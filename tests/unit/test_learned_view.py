"""Unit tests: the learned raaga view excludes scale runs from its idiom
evidence (raagacomposer/agent/learned.py, ``_is_scale_run`` /
``SCALE_RUN_MIN``).

A monotone run through six or more consecutive scale degrees - the arohanam
or avarohanam as actually played - is a fact the raaga already carries, not
a characteristic phrase, so it is kept out of ``prayogas`` (what
``_phrase_tokens`` may quote) and out of the idiom's evidence (what
``RaagaIdiom.from_phrases`` learns habits from).  It stays in the wider
phrase bank the evaluator and the originality checker read
(``learned_phrase_bank``), because a tune that plays the scale straight is
still something the agent should recognise as its own if asked, and
something originality should still guard against repeating.
"""
from __future__ import annotations

import pytest

from raagacomposer.agent.idiom import MIN_PHRASES
from raagacomposer.agent.knowledge import KnowledgeRepository, Phrase
from raagacomposer.agent.learned import (SCALE_RUN_MIN, learned_phrase_bank,
                                         learned_raaga)
from raagacomposer.raaga.library import RaagaLibrary

pytestmark = pytest.mark.unit

RAAGA = "Keeravani"


@pytest.fixture
def repo(tmp_path):
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


def _add(repo: KnowledgeRepository, swaras, confidence: float) -> Phrase:
    phrase, _ = repo.add_phrase(Phrase(raaga=RAAGA, swaras=list(swaras),
                                       confidence=confidence))
    return phrase


# The eight-note ascending run: Keeravani's own arohanam, played straight.
SCALE_RUN = ["S", "R2", "G2", "M1", "P", "D1", "N3", "S+"]
# Three ordinary four-note phrases: below SCALE_RUN_MIN, so never excluded
# regardless of shape - the exclusion is about length, not direction.
SHORT_PHRASES = [
    ["S", "R2", "G2", "M1"],
    ["P", "D1", "N3", "S+"],
    ["M1", "G2", "R2", "S"],
]


def test_scale_run_is_excluded_from_prayogas_and_idiom(repo, raagas: RaagaLibrary):
    assert len(SCALE_RUN) >= SCALE_RUN_MIN
    _add(repo, SCALE_RUN, confidence=0.97)
    for phrase in SHORT_PHRASES:
        _add(repo, phrase, confidence=0.8)

    raaga, _ = learned_raaga(repo, raagas, RAAGA)
    assert raaga is not None

    assert SCALE_RUN not in raaga.prayogas
    for phrase in SHORT_PHRASES:
        assert phrase in raaga.prayogas

    assert raaga.idiom is not None
    assert raaga.idiom.phrases == len(SHORT_PHRASES) == 3


def test_scale_run_still_reaches_the_wider_phrase_bank(repo, raagas: RaagaLibrary):
    """Excluded from the idiom and from quoting, but not erased: the
    evaluator's authenticity check and the originality checker both read
    ``learned_phrase_bank``, and a played-straight scale is still something
    the agent heard and should recognise."""
    _add(repo, SCALE_RUN, confidence=0.97)
    for phrase in SHORT_PHRASES:
        _add(repo, phrase, confidence=0.8)

    bank = learned_phrase_bank(repo, RAAGA)
    assert SCALE_RUN in bank
    for phrase in SHORT_PHRASES:
        assert phrase in bank
    assert len(bank) == 4


def test_a_five_note_run_is_below_the_scale_run_floor(repo, raagas: RaagaLibrary):
    """SCALE_RUN_MIN is 6: a run one note short of it is an ordinary short
    phrase, not "the scale as played", and is not excluded."""
    five_note_run = ["S", "R2", "G2", "M1", "P"]
    assert len(five_note_run) == SCALE_RUN_MIN - 1
    _add(repo, five_note_run, confidence=0.9)
    for phrase in SHORT_PHRASES:
        _add(repo, phrase, confidence=0.8)

    raaga, _ = learned_raaga(repo, raagas, RAAGA)
    assert raaga is not None
    assert five_note_run in raaga.prayogas


def test_a_run_with_one_repeated_note_is_not_excluded(repo, raagas: RaagaLibrary):
    """A run of six or more notes that is not strictly monotone - here, one
    note repeats instead of stepping - is not "the scale as played" and is
    kept as a genuine phrase and idiom evidence."""
    almost_scale_run = ["S", "R2", "G2", "G2", "M1", "P"]
    assert len(almost_scale_run) >= SCALE_RUN_MIN
    _add(repo, almost_scale_run, confidence=0.9)
    for phrase in SHORT_PHRASES:
        _add(repo, phrase, confidence=0.8)

    raaga, _ = learned_raaga(repo, raagas, RAAGA)
    assert raaga is not None
    assert almost_scale_run in raaga.prayogas
    assert raaga.idiom is not None
    assert raaga.idiom.phrases == 1 + len(SHORT_PHRASES)


def test_min_phrases_still_applies_after_exclusion(repo, raagas: RaagaLibrary):
    """Excluding the scale run can drop the heard-phrase count below
    ``MIN_PHRASES``, in which case there is still no idiom - the exclusion
    happens before the idiom is built, not after."""
    _add(repo, SCALE_RUN, confidence=0.97)
    _add(repo, SHORT_PHRASES[0], confidence=0.8)
    assert 1 < MIN_PHRASES

    raaga, _ = learned_raaga(repo, raagas, RAAGA)
    assert raaga is not None
    assert SCALE_RUN not in raaga.prayogas
    assert raaga.idiom is None
