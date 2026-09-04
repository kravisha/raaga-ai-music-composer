"""Unit tests: the raaga idiom model (agent/idiom.py).

``RaagaIdiom.from_phrases`` is a pure function of a phrase list, so most of
this file constructs ``Phrase`` rows directly rather than going through a
repository or the agent.
"""
from __future__ import annotations

import random

import pytest

from raagacomposer.agent.idiom import MIN_PHRASES, PRIOR, PRIOR_WEIGHT, RaagaIdiom
from raagacomposer.agent.knowledge import Phrase
from raagacomposer.agent.originality import DEFAULT_MAX_RUN
from raagacomposer.music.melody import MAX_QUOTE_NOTES

pytestmark = pytest.mark.unit


def _phrase(swaras, confidence=0.8):
    return Phrase(raaga="Keeravani", swaras=list(swaras), confidence=confidence)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------
def test_from_phrases_on_keeravanis_own_prayogas(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    assert idiom.phrases == len(phrases)

    # A transition row exists for every base swara that starts a move in any
    # heard phrase.
    from raagacomposer.raaga.library import parse_swara
    starting_bases = set()
    for p in keeravani.prayogas:
        starting_bases.update(parse_swara(t)[0] for t in p[:-1])
    assert starting_bases <= set(idiom.transitions.keys())

    # Endings sum to about 1 (a probability distribution over final notes).
    assert idiom.endings
    assert sum(idiom.endings.values()) == pytest.approx(1.0, abs=0.01)

    # Contours were classified for every phrase.
    assert idiom.contours
    assert sum(idiom.contours.values()) == pytest.approx(1.0, abs=0.01)

    # Mean length matches the phrase lengths actually fed in - not a
    # hardcoded number, since the shipped prayoga lengths could change.
    expected_mean = sum(len(p) for p in keeravani.prayogas) / len(keeravani.prayogas)
    assert idiom.mean_length == pytest.approx(expected_mean, abs=1e-3)


def test_fewer_than_min_phrases_gives_none(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas[:MIN_PHRASES - 1]]
    assert len(phrases) < MIN_PHRASES
    assert RaagaIdiom.from_phrases(keeravani, phrases) is None


def test_exactly_min_phrases_is_enough(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas[:MIN_PHRASES]]
    assert RaagaIdiom.from_phrases(keeravani, phrases) is not None


def test_confidence_weights_endings(keeravani):
    """A phrase heard with high confidence outweighs one heard with low
    confidence when it comes to what the raaga is said to rest on."""
    strong = _phrase(["S", "R2", "G2", "M1", "P"], confidence=0.9)   # ends P
    weak = _phrase(["S", "R2", "G2", "M1"], confidence=0.1)          # ends M1
    filler = [_phrase(["P", "D1", "N3", "S+"], confidence=0.5) for _ in range(2)]
    idiom = RaagaIdiom.from_phrases(keeravani, [strong, weak] + filler)
    assert idiom is not None
    # P is bolstered by both the strong phrase and the filler; M1 only by
    # the weak one, so P's share must clearly exceed M1's.
    assert idiom.endings.get("P", 0.0) > idiom.endings.get("M1", 0.0) * 3


# --------------------------------------------------------------------------
# the composer's own habits are a prior, not a blank slate
# --------------------------------------------------------------------------
def test_one_heard_phrase_shades_the_prior_without_overriding_it(keeravani):
    """A single heard transition (S -> R2, an ``up1`` move) is folded into
    the composer's own habits as pseudo-observations rather than replacing
    them: the resulting weight is the one observation plus the prior's own
    share, and every *other* move at that note keeps exactly the prior's
    shape, scaled by ``PRIOR_WEIGHT``."""
    phrases = [
        _phrase(["S", "R2"], confidence=1.0),          # S -> R2: up1
        _phrase(["P", "D1", "N3"], confidence=1.0),     # does not touch S
        _phrase(["M1", "G2"], confidence=1.0),          # does not touch S
    ]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None

    weights = idiom.move_weights("S")
    assert weights["up1"] == pytest.approx(1.0 + PRIOR["up1"] * PRIOR_WEIGHT)
    assert weights["up1"] == pytest.approx(1.2 + 1.0)
    assert weights["same"] == pytest.approx(PRIOR["same"] * PRIOR_WEIGHT)
    assert weights["same"] == pytest.approx(0.8)
    for move in ("up2", "down1", "down2"):
        assert weights[move] == pytest.approx(PRIOR[move] * PRIOR_WEIGHT)


def test_pick_direction_never_collapses_to_one_outcome(keeravani):
    """Even with every heard transition from a note pointing the same way,
    the prior keeps the other two directions reachable: over many draws each
    of +1, -1 and 0 comes up at least some of the time."""
    phrases = [
        _phrase(["S", "R2"], confidence=1.0),
        _phrase(["P", "D1", "N3"], confidence=1.0),
        _phrase(["M1", "G2"], confidence=1.0),
    ]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None

    rng = random.Random(99)
    n = 2000
    counts = {1: 0, -1: 0, 0: 0}
    for _ in range(n):
        counts[idiom.pick_direction("S", rng)] += 1
    for outcome, count in counts.items():
        assert count >= 0.05 * n, (outcome, count, counts)


# --------------------------------------------------------------------------
# reading: exactly one rng draw
# --------------------------------------------------------------------------
def test_pick_direction_takes_exactly_one_draw(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None

    rng_a = random.Random(123)
    rng_b = random.Random(123)
    idiom.pick_direction("G2", rng_a)
    rng_b.random()
    assert rng_a.getstate() == rng_b.getstate()


def test_pick_steps_takes_exactly_one_draw(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None

    rng_a = random.Random(7)
    rng_b = random.Random(7)
    idiom.pick_steps("P", 1, rng_a)
    rng_b.random()
    assert rng_a.getstate() == rng_b.getstate()


def test_pick_direction_returns_a_valid_move(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    rng = random.Random(1)
    for _ in range(50):
        assert idiom.pick_direction("S", rng) in (1, -1, 0)


def test_pick_steps_returns_one_or_two(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    rng = random.Random(1)
    for direction in (1, -1):
        for _ in range(50):
            assert idiom.pick_steps("G2", direction, rng) in (1, 2)


# --------------------------------------------------------------------------
# cadence
# --------------------------------------------------------------------------
def test_cadence_for_prefers_the_nyasa_heard_most(keeravani):
    # All phrases end on P; the idiom should say so decisively.
    phrases = [_phrase(["S", "R2", "G2", "M1", "P"]) for _ in range(4)]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    assert max(idiom.endings, key=idiom.endings.get) == "P"

    cadence = idiom.cadence_for(keeravani, "R2")
    from raagacomposer.raaga.library import parse_swara
    base, _ = parse_swara(cadence)
    assert base == "P"


def test_cadence_lands_within_an_octave_of_the_token(keeravani):
    phrases = [_phrase(["S", "R2", "G2", "M1", "P"]) for _ in range(4)]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    ladder_len = len(keeravani.ascending)
    for token in ("S", "G2", "P", "N3", "S+"):
        cadence = idiom.cadence_for(keeravani, token)
        assert abs(keeravani.degree(cadence) - keeravani.degree(token)) <= ladder_len


def test_cadence_for_with_no_nyasa_returns_the_token(keeravani):
    from dataclasses import replace
    bare = replace(keeravani, nyasa=[])
    phrases = [_phrase(["S", "R2", "G2", "M1", "P"]) for _ in range(4)]
    idiom = RaagaIdiom.from_phrases(bare, phrases)
    assert idiom is not None
    assert idiom.cadence_for(bare, "R2") == "R2"


def test_describe_is_non_empty(keeravani):
    phrases = [_phrase(p) for p in keeravani.prayogas]
    idiom = RaagaIdiom.from_phrases(keeravani, phrases)
    assert idiom is not None
    text = idiom.describe()
    assert text
    assert text != "no idiom"


def test_describe_of_no_idiom():
    empty = RaagaIdiom()
    assert empty.describe() == "no idiom"


# --------------------------------------------------------------------------
# the two limits agree without an import across the music/agent boundary
# --------------------------------------------------------------------------
def test_max_quote_notes_matches_the_originality_limit():
    assert MAX_QUOTE_NOTES == DEFAULT_MAX_RUN
