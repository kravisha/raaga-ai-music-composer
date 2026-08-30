"""Unit tests: syllable fitting and lyric generation."""
from __future__ import annotations

import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.core.versioning import LockedContentError
from raagacomposer.lyrics.fitting import (alignment_report, build_slots,
                                          count_syllables, fit_line, fit_lines,
                                          refit_line, split_line_syllables,
                                          syllabify)
from raagacomposer.lyrics.generator import (generate, generate_lines, make_line,
                                            regenerate_line)
from raagacomposer.music.melody import MelodyOptions, generate as gen_melody

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def melody(request):
    from raagacomposer.raaga.library import library
    raaga = library().require("Charukesi")
    return gen_melody(raaga, MelodyOptions(tempo_bpm=68, seed=7,
                                           duration_target=120))


# --------------------------------------------------------------------------
# syllabification
# --------------------------------------------------------------------------
@pytest.mark.parametrize("word,count", [
    ("kaadhal", 2),
    ("nilavu", 3),
    ("iravu", 3),
    ("pyaar", 1),
    ("a", 1),
])
def test_syllable_counts(word, count):
    assert len(syllabify(word)) == count


def test_syllabify_ignores_punctuation_and_empty_input():
    assert syllabify("") == []
    assert syllabify("...") == []
    assert syllabify("nee,") == syllabify("nee")


def test_count_and_split_agree():
    text = "kaadhal iravu nilavu"
    assert count_syllables(text) == len(split_line_syllables(text))


# --------------------------------------------------------------------------
# slots
# --------------------------------------------------------------------------
def test_slots_only_cover_sung_sections(melody):
    slots = build_slots(melody)
    assert slots
    instrumental = {s.id for s in melody.sections if s.kind.instrumental}
    assert all(slot.section_id not in instrumental for slot in slots)


def test_slot_reports_its_syllable_count_and_stresses(melody):
    slot = build_slots(melody)[0]
    assert slot.syllable_count == len(slot.note_indices)
    assert len(slot.stresses) == slot.syllable_count
    assert slot.stresses[0] is True          # phrase openings carry weight
    assert slot.end > slot.start
    assert "syllables" in slot.describe()


def test_instrumental_slots_can_be_requested(melody):
    with_instrumental = build_slots(melody, include_instrumental=True)
    assert len(with_instrumental) > len(build_slots(melody))


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def test_exact_fit_maps_one_syllable_per_note(melody):
    slot = build_slots(melody)[0]
    text = " ".join(["la"] * slot.syllable_count)
    syllables, notes, warnings = fit_line(text, slot)
    assert len(syllables) == len(notes) == slot.syllable_count
    assert not warnings


def test_short_line_produces_melisma(melody):
    slot = next(s for s in build_slots(melody) if s.syllable_count >= 4)
    syllables, notes, warnings = fit_line("nilavu", slot)
    assert len(syllables) == len(notes)
    assert any(s.startswith("~") for s in syllables)
    assert not warnings


def test_long_line_packs_syllables_and_warns(melody):
    slot = build_slots(melody)[0]
    text = " ".join(["kaadhal"] * (slot.syllable_count + 2))
    syllables, notes, warnings = fit_line(text, slot)
    assert len(syllables) == len(notes)
    assert warnings and "doubled up" in warnings[0]


def test_empty_line_is_reported(melody):
    slot = build_slots(melody)[0]
    _, _, warnings = fit_line("", slot)
    assert warnings


def test_fit_lines_builds_one_line_per_slot(melody):
    slots = build_slots(melody)
    lyrics = fit_lines(["la la"] * len(slots), melody, "Tamil", version=1)
    assert len(lyrics.lines) == len(slots)
    assert lyrics.language == "Tamil"
    assert lyrics.melody_version == melody.version
    for line, slot in zip(lyrics.lines, slots):
        assert line.note_indices == slot.note_indices
        assert line.start == pytest.approx(slot.start)


def test_refit_line_touches_only_that_line(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=3)
    others = [(l.id, l.text, tuple(l.syllables)) for l in lyrics.lines[1:]]
    refit_line(lyrics, melody, lyrics.lines[0].id, "puthiya vaanam")
    assert lyrics.lines[0].text == "puthiya vaanam"
    assert [(l.id, l.text, tuple(l.syllables)) for l in lyrics.lines[1:]] == others


def test_refit_respects_a_locked_line(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=3)
    lyrics.lines[0].locked = True
    with pytest.raises(LockedContentError):
        refit_line(lyrics, melody, lyrics.lines[0].id, "something else")


def test_fit_lines_carries_locked_lines_forward(melody):
    first = generate(melody, CreativeBrief(language="Tamil"), seed=3)
    first.lines[0].locked = True
    kept = first.lines[0].text
    second = fit_lines(["la la"] * len(first.lines), melody, "Tamil",
                       version=2, previous=first)
    assert second.lines[0].text == kept


def test_alignment_report_lists_every_line(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=3)
    report = alignment_report(lyrics, melody)
    assert len(report.splitlines()) == len(lyrics.lines)
    assert "MISFIT" not in report


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def test_make_line_hits_the_exact_syllable_count():
    import random
    words = ["kaadhal", "nilavu", "iravu", "nee", "oru"]
    for target in range(1, 12):
        line = make_line(target, words, random.Random(target))
        assert count_syllables(line) == target, (target, line)


@pytest.mark.parametrize("language", ["Tamil", "Hindi", "Telugu", "English"])
def test_generated_lyrics_fit_every_language(melody, language):
    lyrics = generate(melody, CreativeBrief(language=language), seed=4)
    assert lyrics.lines
    for line in lyrics.lines:
        assert len(line.syllables) == len(line.note_indices)
        assert line.text.strip()


def test_generated_lines_match_the_slot_counts(melody):
    slots = build_slots(melody)
    lines = generate_lines(slots, CreativeBrief(language="Tamil"), seed=6)
    assert len(lines) == len(slots)
    for line, slot in zip(lines, slots):
        assert count_syllables(line) == slot.syllable_count


def test_adjacent_phrases_in_a_section_are_not_identical(melody):
    slots = build_slots(melody)
    lines = generate_lines(slots, CreativeBrief(language="Tamil"), seed=6)
    for (a, sa), (b, sb) in zip(zip(lines, slots), zip(lines[1:], slots[1:])):
        if sa.section_id == sb.section_id:
            assert a != b, f"repeated line inside {sa.section_name}"


def test_regenerate_one_line_leaves_the_others(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4)
    others = [(l.id, l.text) for l in lyrics.lines[1:]]
    regenerate_line(lyrics, melody, lyrics.lines[0].id,
                    CreativeBrief(language="Tamil"), seed=99)
    assert [(l.id, l.text) for l in lyrics.lines[1:]] == others
    assert len(lyrics.lines[0].syllables) == len(lyrics.lines[0].note_indices)


def test_generation_with_no_vocal_phrases_reports_it(keeravani):
    melody = gen_melody(keeravani, MelodyOptions(seed=2, duration_target=90))
    melody.notes = []
    lyrics = generate(melody, CreativeBrief(language="Tamil"))
    assert lyrics.lines == []
    assert "no vocal phrases" in lyrics.notes
