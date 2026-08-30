"""Unit tests: melody generation, editing operations and the raaga validator."""
from __future__ import annotations

import pytest

from raagacomposer.core.versioning import LockedContentError
from raagacomposer.music.melody import (MelodyOptions, clamp_token, generate,
                                        notes_in_range, phrase_boundaries,
                                        regenerate_section, retempo, transpose,
                                        variation)
from raagacomposer.music.validator import repair, validate
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.unit


def _opts(**kw) -> MelodyOptions:
    base = dict(tempo_bpm=72, seed=5, duration_target=90, tonic_midi=60,
                voice_low=52, voice_high=79)
    base.update(kw)
    return MelodyOptions(**base)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def test_generation_produces_a_usable_tune(keeravani):
    melody = generate(keeravani, _opts())
    assert melody.notes
    assert melody.sections
    assert melody.duration > 60
    assert melody.raaga == "Keeravani"
    assert all(n.duration > 0 for n in melody.notes)
    assert all(n.section_id for n in melody.notes)


def test_notes_never_leave_the_singer_range(keeravani):
    opts = _opts(voice_low=55, voice_high=76)
    melody = generate(keeravani, opts)
    assert all(55 <= n.midi <= 76 for n in melody.notes)


def test_notes_are_ordered_and_do_not_overlap(keeravani):
    melody = generate(keeravani, _opts())
    for a, b in zip(melody.notes, melody.notes[1:]):
        assert b.start >= a.start
        # Starts and durations are stored rounded to 4 decimals, so allow a
        # rounding epsilon rather than an exact non-overlap.
        assert b.start >= a.end - 1e-3


def test_generation_is_deterministic_for_a_seed(keeravani):
    a = generate(keeravani, _opts(seed=42))
    b = generate(keeravani, _opts(seed=42))
    assert [(n.swara, n.start, n.duration) for n in a.notes] == \
           [(n.swara, n.start, n.duration) for n in b.notes]


def test_different_seeds_give_different_tunes(keeravani):
    a = generate(keeravani, _opts(seed=1))
    b = generate(keeravani, _opts(seed=2))
    assert [n.swara for n in a.notes] != [n.swara for n in b.notes]


def test_every_section_gets_notes(keeravani):
    melody = generate(keeravani, _opts())
    for section in melody.sections:
        assert any(n.section_id == section.id for n in melody.notes), section.name


def test_gamaka_marks_come_from_the_raaga(keeravani):
    melody = generate(keeravani, _opts())
    marked = [n for n in melody.notes if n.gamaka]
    assert marked, "no gamaka was applied at all"
    for note in marked:
        base = parse_swara(note.swara)[0]
        assert keeravani.gamaka.get(base) == note.gamaka


def test_phrases_are_separated_by_breaths(keeravani):
    melody = generate(keeravani, _opts())
    phrases = phrase_boundaries(melody)
    assert len(phrases) > 1
    assert sum(len(p) for p in phrases) == len(melody.notes)


def test_clamp_token_keeps_swara_and_pitch_consistent(keeravani):
    token = clamp_token(keeravani, "S", 60, 70, 84)
    assert keeravani.midi(token, 60) >= 70
    assert parse_swara(token)[0] == "S"


def test_notes_in_range_selects_by_overlap(keeravani):
    melody = generate(keeravani, _opts())
    window = notes_in_range(melody, 20.0, 30.0)
    assert window
    assert all(n.start < 30.0 and n.end > 20.0 for n in window)


# --------------------------------------------------------------------------
# editing operations
# --------------------------------------------------------------------------
def test_regenerating_one_section_leaves_the_rest_bit_identical(keeravani):
    melody = generate(keeravani, _opts())
    target = melody.sections[1]
    before = [(n.start, n.midi, n.swara) for n in melody.notes
              if n.section_id != target.id]
    fresh = regenerate_section(melody, keeravani, target.id, _opts(), 2)
    after = [(n.start, n.midi, n.swara) for n in fresh.notes
             if n.section_id != target.id]
    assert before == after
    assert fresh.version == 2
    assert fresh.parent_version == melody.version
    assert target.name in fresh.derived_from


def test_regenerating_a_locked_section_is_refused(keeravani):
    melody = generate(keeravani, _opts())
    melody.sections[1].locked = True
    with pytest.raises(LockedContentError):
        regenerate_section(melody, keeravani, melody.sections[1].id, _opts(), 2)


def test_variation_preserves_locked_sections(keeravani):
    melody = generate(keeravani, _opts())
    locked = melody.sections[1]
    locked.locked = True
    kept = [(n.start, n.midi) for n in melody.notes if n.section_id == locked.id]
    fresh = variation(melody, keeravani, _opts(), 2)
    after = [(n.start, n.midi) for n in fresh.notes if n.section_id == locked.id]
    assert kept == after
    changed = [n.swara for n in fresh.notes if n.section_id == melody.sections[0].id]
    original = [n.swara for n in melody.notes if n.section_id == melody.sections[0].id]
    assert changed != original


def test_retempo_keeps_the_pitches_and_scales_the_clock(keeravani):
    melody = generate(keeravani, _opts(tempo_bpm=80))
    faster = retempo(melody, 120, 2)
    assert [n.midi for n in faster.notes] == [n.midi for n in melody.notes]
    assert faster.tempo_bpm == 120
    assert faster.duration == pytest.approx(melody.duration * 80 / 120, rel=0.01)
    assert faster.sections[-1].end == pytest.approx(
        melody.sections[-1].end * 80 / 120, rel=0.01)


def test_transpose_shifts_every_note_equally(keeravani):
    melody = generate(keeravani, _opts())
    up = transpose(melody, 3, 2)
    assert up.tonic_midi == melody.tonic_midi + 3
    assert all(b.midi - a.midi == 3 for a, b in zip(melody.notes, up.notes))


def test_edits_never_mutate_the_source_version(keeravani):
    melody = generate(keeravani, _opts())
    snapshot = [(n.start, n.midi) for n in melody.notes]
    regenerate_section(melody, keeravani, melody.sections[0].id, _opts(), 2)
    variation(melody, keeravani, _opts(), 3)
    retempo(melody, 100, 4)
    assert [(n.start, n.midi) for n in melody.notes] == snapshot


# --------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Kalyani", "Keeravani", "Mohanam", "Abheri",
                                  "Hamsadhwani", "Bhairavi", "Hindolam"])
def test_generated_tunes_pass_the_raaga_check(raagas, name):
    raaga = raagas.require(name)
    melody = generate(raaga, _opts(seed=11))
    report = validate(melody, raaga, 52, 79)
    assert report.stats["out_of_raaga"] == 0
    assert report.score >= 0.9, report.summary()


def test_validator_flags_a_note_outside_the_raaga(mohanam):
    melody = generate(mohanam, _opts())
    melody.notes[3].swara = "M1"          # Mohanam has no madhyamam
    melody.notes[3].midi = melody.tonic_midi + 5
    report = validate(melody, mohanam, 52, 79)
    assert not report.ok
    assert report.stats["out_of_raaga"] == 1
    assert "outside Mohanam" in report.summary()


def test_validator_flags_notes_outside_the_voice_range(keeravani):
    melody = generate(keeravani, _opts())
    melody.notes[0].midi = 20
    report = validate(melody, keeravani, 52, 79)
    assert any("voice range" in issue for issue in report.issues)


def test_validator_handles_an_empty_tune(keeravani):
    melody = generate(keeravani, _opts())
    melody.notes = []
    report = validate(melody, keeravani)
    assert report.score == 0.0
    assert not report.ok


def test_repair_snaps_stray_notes_back_into_the_raaga(mohanam):
    melody = generate(mohanam, _opts())
    melody.notes[2].swara = "M1"
    melody.notes[2].midi = melody.tonic_midi + 5
    fixed = repair(melody, mohanam)
    assert fixed == 1
    assert validate(melody, mohanam, 52, 79).stats["out_of_raaga"] == 0
