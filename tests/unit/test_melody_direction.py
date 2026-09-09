"""A direction has a measurable effect through the real rewrite path.

Each case rewrites one section of a fixed-seed tune twice - once with the
direction, once without (the lever disconnected) - and measures the
section's notes.  The undirected rewrite is the control; the same seed
makes the two comparable.
"""
from dataclasses import replace

import pytest

from raagacomposer.agent.guidance import Guidance
from raagacomposer.music.direction import apply_direction, read_direction
from raagacomposer.music.melody import (MelodyOptions, generate, regenerate_section)
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.unit

SEEDS = (1, 2, 3, 4, 5)


def _opts(seed):
    return MelodyOptions(tempo_bpm=72, seed=seed, duration_target=90,
                         tonic_midi=60, voice_low=52, voice_high=79)


def _tune(raaga, seed):
    return generate(raaga, _opts(seed))


def _charanam(melody):
    return next(s for s in melody.sections if s.name == "Charanam 1")


def _rewrite(melody, raaga, text, seed, lessons=None):
    """(directed section notes, undirected section notes) for one seed."""
    section = _charanam(melody)
    base = lessons if lessons is not None else Guidance()
    plain = _opts(seed + 100)
    plain.guidance = base
    directed = apply_direction(read_direction(text), _opts(seed + 100), base, raaga=raaga)
    directed.opts.guidance = directed.guidance
    without = regenerate_section(melody, raaga, section.id, plain, 2)
    with_it = regenerate_section(melody, raaga, section.id, directed.opts, 2)
    pick = lambda m: [n for n in m.notes if n.section_id == section.id]  # noqa: E731
    return pick(with_it), pick(without), with_it, without


def _mean(xs):
    return sum(xs) / max(1, len(xs))


def _intervals(notes):
    return [abs(b.midi - a.midi) for a, b in zip(notes, notes[1:])]


def test_softer_lowers_the_velocity(keeravani):
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "softer", seed)
        assert _mean([n.velocity for n in with_it]) < _mean([n.velocity for n in without]), seed


def test_stronger_raises_the_velocity(keeravani):
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "stronger", seed)
        assert _mean([n.velocity for n in with_it]) > _mean([n.velocity for n in without]), seed


def test_less_ornament_wins_over_a_lesson_that_favoured_gamaka(keeravani):
    lessons = Guidance(add_gamaka=True)
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "plainer",
                                          seed, lessons=lessons)
        ornamented = sum(1 for n in with_it if n.gamaka)
        control = sum(1 for n in without if n.gamaka)
        assert control >= 1, "the control had gamaka to lose"
        assert ornamented < control, (seed, ornamented, control)
    assert lessons.add_gamaka is True, "the lesson guidance was changed"


def test_more_ornament_adds_gamaka(keeravani):
    total_with, total_without = 0, 0
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "more gamaka", seed)
        total_with += sum(1 for n in with_it if n.gamaka)
        total_without += sum(1 for n in without if n.gamaka)
    assert total_with > total_without


def test_a_lower_register_moves_the_section_down_and_nowhere_else(keeravani):
    for seed in SEEDS:
        tune = _tune(keeravani, seed)
        with_it, without, directed, plain = _rewrite(tune, keeravani, "a lower register", seed)
        assert _mean([n.midi for n in with_it]) < _mean([n.midi for n in without]), seed
        assert max(n.midi for n in with_it) <= max(n.midi for n in without)
        assert min(n.midi for n in with_it) >= 52
        # The other sections are the tune's own, note for note.
        charanam = _charanam(tune).id
        keep = lambda m: [(n.section_id, n.midi, n.start) for n in m.notes  # noqa: E731
                          if n.section_id != charanam]
        assert keep(directed) == keep(tune) == keep(plain)


def test_a_closer_register_narrows_the_section(keeravani):
    """The reach of each phrase is halved.  A walk is random, so one seed
    can still land wide; the measure is over the seeds together, and the
    span never widens by more than a step on any one."""
    spans_with, spans_without = [], []
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "a closer register", seed)
        spans_with.append(max(n.midi for n in with_it) - min(n.midi for n in with_it))
        spans_without.append(max(n.midi for n in without) - min(n.midi for n in without))
    assert _mean(spans_with) < _mean(spans_without), (spans_with, spans_without)
    assert sum(a < b for a, b in zip(spans_with, spans_without)) >= 3, (spans_with, spans_without)
    assert all(a <= b + 2 for a, b in zip(spans_with, spans_without)), (spans_with, spans_without)


def test_a_resting_cadence_ends_on_a_resting_note(keeravani):
    assert keeravani.nyasa
    for seed in SEEDS:
        with_it, _, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "a clearer cadence", seed)
        assert parse_swara(with_it[-1].swara)[0] in keeravani.nyasa, (seed, with_it[-1].swara)


def test_stepwise_reduces_leaps_and_keeps_the_lessons_forbidden_move(keeravani):
    """The step preference is a probability nudge on a random walk, so it
    is measured over the seeds together; the lesson's forbidden move is a
    hard rule and is checked on every seed - in the control too, since the
    cadence used to overwrite the last note unchecked."""
    lessons = Guidance(avoid_transitions={("P", "S")})
    directed_total, control_total = 0.0, 0.0
    for seed in SEEDS:
        with_it, without, _, _ = _rewrite(_tune(keeravani, seed), keeravani, "stepwise",
                                          seed, lessons=lessons)
        directed_total += _mean(_intervals(with_it))
        control_total += _mean(_intervals(without))
        for notes in (with_it, without):
            moves = [(parse_swara(a.swara)[0], parse_swara(b.swara)[0])
                     for a, b in zip(notes, notes[1:])]
            assert ("P", "S") not in moves, seed
    assert directed_total < control_total, (directed_total, control_total)


def test_an_undirected_rewrite_is_untouched_by_the_change(keeravani):
    """The lever disconnected: an ordinary rewrite is byte-identical to one
    made with no direction support at all."""
    for seed in SEEDS[:2]:
        tune = _tune(keeravani, seed)
        section = _charanam(tune)
        a = regenerate_section(tune, keeravani, section.id, _opts(seed + 100), 2)
        opts = _opts(seed + 100)
        opts.direction = None
        b = regenerate_section(tune, keeravani, section.id, opts, 2)
        assert [(n.swara, n.midi, n.start, n.velocity, n.gamaka) for n in a.notes] == \
            [(n.swara, n.midi, n.start, n.velocity, n.gamaka) for n in b.notes]
