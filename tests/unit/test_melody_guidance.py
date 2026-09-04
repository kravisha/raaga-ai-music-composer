"""Unit tests: guidance in the composer (docs/PLAN_learning_loop.md item 4,
docs/PLAN_agent_factory.md "Item 4, integrated") - ``music/melody.py``
consulting an ``agent.guidance.Guidance`` the same way ``agent/practice.py``
already does.

Two things matter here.  First, with no guidance the draw sequence is
byte-identical to a build with no guidance support at all - the hard
constraint the whole increment is built under.  Second, once guidance is
attached, each of its fields has a measurable effect on what comes out:
``avoid_transitions``/``avoid_runs`` are never taken in the free walk (the
committed tokens of a *quoted* fragment are exempt by design, so those note
ranges - recorded in ``melody.provenance`` - are excluded from the checks
below), ``avoid_endings``/``must_end_on_nyasa``/``prefer_jeeva`` steer the
cadence, and ``add_gamaka`` marks long notes.  Provenance itself is checked
last: every entry points at the notes it describes and its swaras match.
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from raagacomposer.agent.guidance import Guidance
from raagacomposer.music.melody import (MelodyOptions, _cadence_with_guidance,
                                        _phrase_tokens, generate)
from raagacomposer.music.theory import beat_seconds
from raagacomposer.raaga.library import RaagaLibrary, parse_swara

pytestmark = pytest.mark.unit

SEEDS = range(1, 21)


def _opts(seed: int, guidance=None) -> MelodyOptions:
    return MelodyOptions(tempo_bpm=72, seed=seed, duration_target=90,
                         tonic_midi=60, voice_low=52, voice_high=79,
                         guidance=guidance)


def _bases(melody) -> list:
    return [parse_swara(n.swara)[0] for n in melody.notes]


# --------------------------------------------------------------------------
# parity: no guidance changes nothing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_guidance_none_and_an_empty_guidance_object_are_byte_identical(
        raagas: RaagaLibrary, raaga_name):
    """``guidance=None`` (the default), an explicit ``None``, and an empty
    ``Guidance()`` must all reproduce the exact same notes over seeds 1..20 -
    ``_guidance_or_none`` collapses all three onto the same no-op code path,
    so this is the guarantee the golden files (tests/regression/
    test_golden_output.py) are pinning at the byte level."""
    raaga = raagas.require(raaga_name)
    for seed in SEEDS:
        default = generate(raaga, MelodyOptions(tempo_bpm=72, seed=seed,
                                                 duration_target=90,
                                                 tonic_midi=60, voice_low=52,
                                                 voice_high=79))
        explicit_none = generate(raaga, _opts(seed, guidance=None))
        empty = generate(raaga, _opts(seed, guidance=Guidance()))
        key = lambda m: [(n.swara, n.start, n.duration, n.gamaka, n.velocity)
                         for n in m.notes]
        assert key(default) == key(explicit_none) == key(empty), seed


# --------------------------------------------------------------------------
# each field has a visible effect
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_avoid_transitions_never_appears_in_the_free_walk(raagas: RaagaLibrary,
                                                           raaga_name):
    # Isolated from quoting (which is exempt from transition-level guidance
    # by design: a committed quote's internal notes are checked only against
    # ``replays``, not ``allows_transition``) by stripping the phrase bank.
    raaga = replace(raagas.require(raaga_name), prayogas=[])

    target = None
    for seed in range(1, 40):
        rng = random.Random(seed)
        tokens = _phrase_tokens(raaga, rng, "S", 14, "wave", 4.0, 0.3, None)
        bases = [parse_swara(t)[0] for t in tokens]
        for a, b in zip(bases, bases[1:]):
            if a != b:
                target = (a, b)
                break
        if target:
            break
    assert target is not None, "no transition ever occurred to ban"

    guidance = Guidance(avoid_transitions={target})
    for seed in range(1, 40):
        rng = random.Random(seed)
        tokens = _phrase_tokens(raaga, rng, "S", 14, "wave", 4.0, 0.3, None,
                                guidance=guidance)
        bases = [parse_swara(t)[0] for t in tokens]
        for a, b in zip(bases, bases[1:]):
            assert (a, b) != target, (seed, tokens)


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_avoid_runs_never_replays_in_the_free_walk(raagas: RaagaLibrary,
                                                    raaga_name):
    raaga = replace(raagas.require(raaga_name), prayogas=[])

    target = None
    for seed in range(1, 40):
        rng = random.Random(seed)
        tokens = _phrase_tokens(raaga, rng, "S", 14, "wave", 4.0, 0.3, None)
        bases = [parse_swara(t)[0] for t in tokens]
        for i in range(len(bases) - 2):
            run = tuple(bases[i:i + 3])
            if len(set(run)) > 1:
                target = run
                break
        if target:
            break
    assert target is not None, "no three-note run ever occurred to ban"

    guidance = Guidance(avoid_runs={target})
    for seed in range(1, 40):
        rng = random.Random(seed)
        tokens = _phrase_tokens(raaga, rng, "S", 14, "wave", 4.0, 0.3, None,
                                guidance=guidance)
        bases = [parse_swara(t)[0] for t in tokens]
        for i in range(len(bases) - 2):
            assert tuple(bases[i:i + 3]) != target, (seed, tokens)


def test_avoid_endings_never_ends_on_the_forbidden_swara(keeravani):
    avoided = parse_swara(keeravani.nyasa[0])[0]
    guidance = Guidance(avoid_endings={avoided}, must_end_on_nyasa=True)
    for cur in list(keeravani.nyasa) + list(keeravani.jeeva) + ["S", "P"]:
        result = _cadence_with_guidance(keeravani, guidance, cur, None)
        assert result is not None
        assert parse_swara(result)[0] != avoided, cur


def test_must_end_on_nyasa_forces_a_cadence_even_when_none_was_drawn(keeravani):
    """The unguided coin flip in ``generate_section_notes`` can decide a
    phrase gets no cadence at all (``cadence=None``); ``must_end_on_nyasa``
    overrules that."""
    guidance = Guidance(must_end_on_nyasa=True)
    for cur in ["R2", "M1", "D1", "N3"]:
        result = _cadence_with_guidance(keeravani, guidance, cur, None)
        assert result is not None
        assert parse_swara(result)[0] in set(keeravani.nyasa)


def test_prefer_jeeva_picks_a_note_that_is_both_jeeva_and_nyasa(keeravani):
    # Keeravani: nyasa = S, P, G2; jeeva = G2, D1, N3 - G2 is both.
    guidance = Guidance(must_end_on_nyasa=True, prefer_jeeva=True)
    result = _cadence_with_guidance(keeravani, guidance, "M1", None)
    assert result is not None
    assert parse_swara(result)[0] == "G2"


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_add_gamaka_marks_long_notes_that_had_none(raagas: RaagaLibrary,
                                                    raaga_name):
    raaga = raagas.require(raaga_name)
    guidance = Guidance(add_gamaka=True)
    hit = False
    for seed in range(1, 8):
        opts = _opts(seed, guidance=guidance)
        melody = generate(raaga, opts)
        beat = beat_seconds(opts.tempo_bpm)
        long_notes = [n for n in melody.notes if n.duration >= beat * 0.5]
        assert long_notes, seed
        assert all(n.gamaka for n in long_notes), (
            seed, [n for n in long_notes if not n.gamaka])
        hit = True
    assert hit


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def _strip_octave(token: str) -> str:
    return token.replace("+", "").replace("-", "")


@pytest.mark.parametrize("raaga_name", ["Keeravani", "Mohanam"])
def test_provenance_entries_point_at_the_notes_they_describe(raagas: RaagaLibrary,
                                                              raaga_name):
    raaga = raagas.require(raaga_name)
    found_any = False
    for seed in SEEDS:
        melody = generate(raaga, _opts(seed))
        for entry in melody.provenance:
            found_any = True
            assert 0 <= entry["start"] <= entry["end"] < len(melody.notes)
            described = melody.notes[entry["start"]:entry["end"] + 1]
            described_swaras = " ".join(_strip_octave(n.swara) for n in described)
            recorded_swaras = " ".join(_strip_octave(s)
                                       for s in entry["swaras"].split())
            assert described_swaras == recorded_swaras, (seed, entry)
            assert entry["source"] in ("prayoga", "learned")
            section = melody.section_by_id(entry["section_id"])
            assert section is not None
            assert all(n.section_id == section.id for n in described)
    assert found_any, "no quote ever fired across seeds 1..20 for either raaga"
