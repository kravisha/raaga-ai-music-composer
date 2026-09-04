"""Unit tests: the melody engine using a raaga's idiom (docs/PLAN_learning_loop.md,
item 3 - ``RaagaIdiom`` consulted from ``music/melody.py``).

Two guarantees matter here.  First, a library raaga - the shape every raaga
is in until the agent has studied it - carries no idiom and composes exactly
as it always has; ``tests/regression/test_golden_output.py`` pins the exact
bytes, this file pins the *invariant* (``raaga.idiom is None``) that makes
that guard meaningful.  Second, once an idiom is attached, generation still
respects the raaga's own rules: every note stays inside its scale, every
section still cadences on a resting note, and a phrase longer than
``MAX_QUOTE_NOTES`` is never quoted beyond ``QUOTE_FRAGMENT_NOTES`` notes.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from raagacomposer.agent.idiom import RaagaIdiom
from raagacomposer.agent.knowledge import Phrase
from raagacomposer.agent.originality import PhraseIndex, check as check_originality
from raagacomposer.music.melody import (MAX_QUOTE_NOTES, QUOTE_FRAGMENT_NOTES,
                                        MelodyOptions, generate)
from raagacomposer.raaga.library import RaagaLibrary, parse_swara

pytestmark = pytest.mark.unit

SEEDS = range(1, 21)


def _opts(seed: int) -> MelodyOptions:
    return MelodyOptions(tempo_bpm=72, seed=seed, duration_target=90,
                         tonic_midi=60, voice_low=52, voice_high=79)


def _bases(melody) -> list:
    return [parse_swara(n.swara)[0] for n in melody.notes]


# --------------------------------------------------------------------------
# a library raaga carries no idiom, and idiom=None is a genuine no-op
# --------------------------------------------------------------------------
def test_every_library_raaga_has_no_idiom(raagas: RaagaLibrary):
    for raaga in raagas.all():
        assert raaga.idiom is None, raaga.name


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_idiom_none_is_a_no_op_for_every_seed(raagas: RaagaLibrary, raaga_name):
    raaga = raagas.require(raaga_name)
    assert raaga.idiom is None
    stripped = replace(raaga, idiom=None)
    for seed in SEEDS:
        a = generate(raaga, _opts(seed))
        b = generate(stripped, _opts(seed))
        assert [(n.swara, n.start, n.duration) for n in a.notes] == \
               [(n.swara, n.start, n.duration) for n in b.notes], seed


# --------------------------------------------------------------------------
# an idiom-bearing raaga still obeys its own scale and cadence rules
# --------------------------------------------------------------------------
def _with_idiom(raaga):
    idiom = RaagaIdiom.from_phrases(raaga, raaga.prayogas)
    assert idiom is not None
    return replace(raaga, idiom=idiom)


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_idiom_generation_stays_inside_the_raaga(raagas: RaagaLibrary, raaga_name):
    learned = _with_idiom(raagas.require(raaga_name))
    allowed = set(learned.allowed)
    for seed in SEEDS:
        melody = generate(learned, _opts(seed))
        bases = _bases(melody)
        assert bases, seed
        assert all(b in allowed for b in bases), (seed, sorted(set(bases) - allowed))


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_idiom_sections_still_cadence_on_a_nyasa(raagas: RaagaLibrary, raaga_name):
    learned = _with_idiom(raagas.require(raaga_name))
    nyasa = set(learned.nyasa)
    assert nyasa
    for seed in SEEDS:
        melody = generate(learned, _opts(seed))
        for section in melody.sections:
            notes = [n for n in melody.notes if n.section_id == section.id]
            if not notes:
                continue
            last_base = parse_swara(notes[-1].swara)[0]
            assert last_base in nyasa, (seed, section.name, notes[-1].swara)


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_idiom_changes_the_tune_for_most_seeds(raagas: RaagaLibrary, raaga_name):
    """The idiom is meant to be heard: for most of 20 seeds the idiom-driven
    tune differs from the coin-flip tune generated from the same phrase bank
    with the idiom stripped back off."""
    base = raagas.require(raaga_name)
    learned = _with_idiom(base)
    no_idiom = replace(learned, idiom=None)
    differing = 0
    for seed in SEEDS:
        with_idiom = generate(learned, _opts(seed))
        without_idiom = generate(no_idiom, _opts(seed))
        if [n.swara for n in with_idiom.notes] != [n.swara for n in without_idiom.notes]:
            differing += 1
    assert differing > len(SEEDS) // 2, (
        f"idiom changed only {differing}/{len(SEEDS)} seeds for {raaga_name}")


# --------------------------------------------------------------------------
# fragment quoting: a long learned phrase is never copied whole
# --------------------------------------------------------------------------
def _longest_shared_run(seq: list, target: list) -> int:
    """Longest run of consecutive positions where ``seq`` and ``target``
    agree, starting anywhere in either.  A direct sliding-window check,
    independent of ``PhraseIndex``'s n-gram floor (``DEFAULT_NGRAM`` = 5),
    which cannot see a match shorter than its own gram size and so cannot
    pin a limit as small as ``QUOTE_FRAGMENT_NOTES`` (3) on its own."""
    best = 0
    for i in range(len(seq)):
        for j in range(len(target)):
            length = 0
            while (i + length < len(seq) and j + length < len(target)
                   and seq[i + length] == target[j + length]):
                length += 1
            best = max(best, length)
    return best


def test_the_quoting_event_itself_never_exceeds_the_fragment_limit(keeravani):
    """The deliberate quoting mechanism in ``_phrase_tokens`` never draws
    more than ``QUOTE_FRAGMENT_NOTES`` consecutive notes from a phrase
    longer than ``MAX_QUOTE_NOTES`` - isolated from any free-walk note that
    happens to border it.

    A whole-tune scan for this bound is not safe to assert as an absolute:
    with only seven notes per octave, the note *immediately before* a quote
    is drawn by an independent free walk, and it can coincidentally be the
    very scale-degree the target phrase has at that position, stretching the
    *apparent* run past QUOTE_FRAGMENT_NOTES even though the quote itself
    inserted exactly the right number of notes.  Measured: with the full
    ascending scale as the eight-note phrase and Keeravani's ordinary
    (non-idiom) free walk, this happens on most of seeds 1..20.  That risk
    belongs to the coarser check_originality/regenerate loop in
    ``generate_tune`` (docs/PLAN_learning_loop.md, "the loop stays as the
    safety net" - see the next test), not to this bound, so it is pinned
    here against the mechanism directly: a rigged generator that always
    quotes, always draws the crafted phrase, and holds its neighbouring
    free-walk step still (direction 0) so it cannot extend the match."""
    from raagacomposer.music.melody import _phrase_tokens

    long_phrase = ["S", "R2", "G2", "M1", "P", "D1", "N3", "S+"]
    long_bases = [parse_swara(t)[0] for t in long_phrase]
    learned = replace(keeravani, prayogas=[long_phrase])

    class AlwaysQuote(random.Random):
        """Always clears the quoting-probability check, always chooses the
        one phrase in the bank, and holds the bordering free-walk step to
        "same note" so it cannot chain onto the quote and inflate the run
        being measured."""

        def random(self):
            return 0.0

        def choice(self, seq):
            seq = list(seq)
            if seq and isinstance(seq[0], list):
                return long_phrase
            return seq[-1]      # the direction tuple's last entry is 0

    for count in range(1 + QUOTE_FRAGMENT_NOTES, 16):
        tokens = _phrase_tokens(learned, AlwaysQuote(1), "S", count, "rise",
                                4.0, 1.0, None)
        bases = [parse_swara(t)[0] for t in tokens]
        run = _longest_shared_run(bases, long_bases)
        assert run <= QUOTE_FRAGMENT_NOTES, (count, tokens, run)
    # and confirm quoting actually fired at least once, rather than the
    # bound holding vacuously because nothing was ever quoted.
    assert run == QUOTE_FRAGMENT_NOTES, (count, tokens, run)


def test_a_long_learned_phrase_passes_the_originality_checker_too(keeravani):
    """The same guarantee, read through the agent's own originality index
    (agent/originality.py) rather than a hand-rolled comparison - the check
    ``generate_tune`` actually runs before a tune is offered."""
    long_phrase = ["S", "R2", "G2", "M1", "P", "D1", "N3", "S+"]
    learned = replace(keeravani, prayogas=[long_phrase] + keeravani.prayogas)

    index = PhraseIndex()
    index.add(Phrase(raaga="Keeravani", swaras=long_phrase, confidence=1.0))

    for seed in SEEDS:
        melody = generate(learned, _opts(seed))
        swaras = [n.swara for n in melody.notes]
        report = check_originality(swaras, index)
        assert report.longest_match <= MAX_QUOTE_NOTES, (seed, report.summary())
