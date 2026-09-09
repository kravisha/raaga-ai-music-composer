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


def _bases(notes):
    return [parse_swara(n.swara)[0] for n in notes]


def test_a_register_direction_never_introduces_a_swara_the_lessons_forbid(keeravani):
    """Arya's P1 on 95d89fe: Keeravani seed 2, Charanam 1, rewrite seed 102,
    avoid N3, 'closer' - the undirected line had no N3 and the directed
    one had two.  Now every seed and every avoided swara: no directed line
    holds a swara its lessons forbid, or the direction says it could not
    be done."""
    found = 0
    for seed in range(1, 8):
        tune = _tune(keeravani, seed)
        section = _charanam(tune)
        for avoided in keeravani.ascending:
            lessons = Guidance(avoid_swaras={avoided})
            plain = _opts(seed + 100)
            plain.guidance = lessons
            control = regenerate_section(tune, keeravani, section.id, plain, 2)
            control_notes = [n for n in control.notes if n.section_id == section.id]
            if avoided in _bases(control_notes):
                continue    # the walk itself broke the lesson; not this test's subject
            for text in ("closer", "a lower register", "a higher register"):
                direction = read_direction(text)
                directed = apply_direction(direction, _opts(seed + 100), lessons, raaga=keeravani)
                directed.opts.guidance = directed.guidance
                out = regenerate_section(tune, keeravani, section.id, directed.opts, 2)
                notes = [n for n in out.notes if n.section_id == section.id]
                assert avoided not in _bases(notes), (seed, avoided, text, [n.swara for n in notes])
                found += 1
    assert found > 20


def test_a_register_direction_keeps_the_lessons_forbidden_moves_out(keeravani):
    lessons = Guidance(avoid_transitions={("P", "S"), ("N3", "S")}, avoid_endings={"N3"})
    for seed in SEEDS:
        tune = _tune(keeravani, seed)
        section = _charanam(tune)
        plain = _opts(seed + 100)
        plain.guidance = lessons
        control = [n for n in regenerate_section(tune, keeravani, section.id, plain, 2).notes
                   if n.section_id == section.id]
        for text in ("closer", "a lower register"):
            direction = read_direction(text)
            directed = apply_direction(direction, _opts(seed + 100), lessons, raaga=keeravani)
            directed.opts.guidance = directed.guidance
            notes = [n for n in regenerate_section(tune, keeravani, section.id, directed.opts, 2).notes
                     if n.section_id == section.id]
            bad = [(a, b) for a, b in zip(_bases(notes), _bases(notes)[1:])
                   if (a, b) in lessons.avoid_transitions]
            control_bad = [(a, b) for a, b in zip(_bases(control), _bases(control)[1:])
                           if (a, b) in lessons.avoid_transitions]
            assert len(bad) <= len(control_bad), (seed, text, bad)
            assert _bases(notes)[-1] != "N3" or _bases(control)[-1] == "N3", (seed, text)


def test_a_register_with_no_allowed_note_inside_it_is_reported_not_forced(keeravani):
    """Every swara but S forbidden: nothing allowed can lie inside a lower
    window that S does not reach, so the notes stay where they were and
    the direction says so."""
    from raagacomposer.core.models import Note
    from raagacomposer.music.melody import _pull_into_window
    others = {s for s in keeravani.ascending if s != "S"}
    lessons = Guidance(avoid_swaras=others)
    notes = [Note(swara=t, midi=keeravani.midi(t, 60), start=i * 0.5, duration=0.5,
                  velocity=80, section_id="x") for i, t in enumerate(["R2", "G2", "M1", "P"])]
    before = [(n.swara, n.midi) for n in notes]
    # 73-83 holds no S (72 and 84 lie outside): nothing allowed is inside,
    # so nothing moves and every note is reported left.
    moved, left = _pull_into_window(keeravani, notes, 60, 73, 83, lessons)
    assert (moved, left) == (0, 4)
    assert [(n.swara, n.midi) for n in notes] == before
    # 44-58 holds S- at 48, the one allowed note: every note lands on it,
    # never on a forbidden neighbour that would have been nearer.
    moved, left = _pull_into_window(keeravani, notes, 60, 44, 58, lessons)
    assert (moved, left) == (4, 0)
    assert all(parse_swara(n.swara)[0] == "S" and n.midi == 48 for n in notes)


def test_what_the_register_repair_could_not_do_reaches_the_direction(keeravani, monkeypatch):
    """Through regenerate_section, the two ways a register can fail are
    both said on the direction: notes with no allowed place inside the
    window, and a repair that would break a lesson (the walk then stands).
    The failures are forced here, since the lessons that force them
    naturally are rare; the wiring is what is checked."""
    from raagacomposer.music import melody as engine
    lessons = Guidance(avoid_swaras={"N3"})
    tune = _tune(keeravani, 3)
    section = _charanam(tune)

    monkeypatch.setattr(engine, "_pull_into_window", lambda *a, **k: (0, 3))
    direction = read_direction("a higher register")
    directed = apply_direction(direction, _opts(103), lessons, raaga=keeravani)
    directed.opts.guidance = directed.guidance
    regenerate_section(tune, keeravani, section.id, directed.opts, 2)
    assert direction.infeasible and "3 note(s) had no allowed place" in direction.infeasible[0]
    assert "could not be done" in direction.describe()

    monkeypatch.undo()
    calls = []
    real = engine._forbidden_moves

    def worse_after(notes, guidance):
        calls.append(1)
        return real(notes, guidance) + (1 if len(calls) > 1 else 0)
    monkeypatch.setattr(engine, "_forbidden_moves", worse_after)
    plain = _opts(103)
    plain.guidance = lessons
    directed_walk = apply_direction(read_direction("a lower register"), _opts(103), lessons,
                                    raaga=keeravani)
    directed_walk.opts.guidance = directed_walk.guidance
    out = regenerate_section(tune, keeravani, section.id, directed_walk.opts, 2)
    assert directed_walk.opts.direction.infeasible
    assert "lessons forbid; left as composed" in directed_walk.opts.direction.infeasible[0]
    assert [n for n in out.notes if n.section_id == section.id]


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
