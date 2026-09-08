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


# --------------------------------------------------------------------------
# text the singer is given is text the creator wrote: Tamil script and
# accented transliteration survive fitting (2026-09-08, Arya's finding)
# --------------------------------------------------------------------------
TAMIL_FIRST_LINE = "திரும்பி வந்தாய், அன்பே"
TAMIL_SLOTS = ["தி", "ரும்", "பி", "வந்", "தாய்", "அன்", "பே"]


def test_tamil_script_syllabifies_by_akshara_with_dead_consonant_as_coda():
    """A consonant with its vowel sign is one syllable; a dead consonant
    (pulli) closes the syllable before it.  The seven slots are the ones
    hand-articulated for the authored first line."""
    assert syllabify("திரும்பி") == ["தி", "ரும்", "பி"]
    assert syllabify("வந்தாய்,") == ["வந்", "தாய்"]
    assert syllabify("அன்பே") == ["அன்", "பே"]
    assert split_line_syllables(TAMIL_FIRST_LINE) == TAMIL_SLOTS
    assert count_syllables(TAMIL_FIRST_LINE) == 7


@pytest.mark.parametrize("accented,plain", [
    ("Praṇaṇa", "Pranana"),
    ("kādhal", "kaadhal"),
    ("nilavē", "nilavee"),
])
def test_accented_transliteration_keeps_its_letters(accented, plain):
    """A diacritic is not punctuation.  The split follows the plain
    spelling; the tokens keep the accented letters the creator wrote."""
    tokens = syllabify(accented)
    assert len(tokens) == len(syllabify(plain)), (tokens, syllabify(plain))
    assert "".join(tokens) == "".join(c for c in accented if c.isalnum() or not c.isascii())


def test_tamil_line_fits_its_notes_and_keeps_its_text(melody):
    from raagacomposer.lyrics.fitting import PhraseSlot
    exact = PhraseSlot(section_id="s", section_name="Pallavi",
                       note_indices=list(range(7)), durations=[0.5] * 7,
                       stresses=[True] + [False] * 6)
    syllables, notes, warnings = fit_line(TAMIL_FIRST_LINE, exact)
    assert syllables == TAMIL_SLOTS and not warnings

    # Two more notes than syllables: two holds, no written syllable consumed.
    longer = PhraseSlot(section_id="s", section_name="Pallavi",
                        note_indices=list(range(9)), durations=[0.5] * 9,
                        stresses=[True] + [False] * 8)
    syllables, notes, warnings = fit_line(TAMIL_FIRST_LINE, longer)
    assert len(syllables) == 9 and not warnings
    assert [s for s in syllables if not s.startswith("~")] == TAMIL_SLOTS
    assert sum(1 for s in syllables if s.startswith("~")) == 2

    # Fewer notes: packed, warned, and still the creator's syllables.
    shorter = PhraseSlot(section_id="s", section_name="Pallavi",
                         note_indices=list(range(5)), durations=[0.5] * 5,
                         stresses=[True] + [False] * 4)
    syllables, notes, warnings = fit_line(TAMIL_FIRST_LINE, shorter)
    assert len(syllables) == 5 and warnings
    assert "".join(syllables) == "".join(TAMIL_SLOTS)

    # And through fit_lines the text itself is verbatim.
    lv = fit_lines([TAMIL_FIRST_LINE] * 3, melody, "Tamil")
    assert lv.lines[0].text == TAMIL_FIRST_LINE


class _Writer:
    """A stand-in language model that answers with given lines."""
    available = True
    name = "test-writer"

    def __init__(self, lines):
        self._lines = lines

    def write_lyrics(self, slots, brief):
        return list(self._lines)


def test_a_writer_s_tamil_lines_are_kept_not_replaced_by_vocables(melody):
    """Native script counted as zero syllables and was replaced one for one
    by lexicon vocables, silently.  The creator's or the writer's words are
    the words; each line records who wrote it."""
    slots = build_slots(melody)
    lines = [TAMIL_FIRST_LINE] * len(slots)
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer(lines))
    assert [l.text for l in lyrics.lines] == lines
    assert all(l.source == "llm:test-writer" for l in lyrics.lines), \
        [l.source for l in lyrics.lines]
    assert all(len(l.syllables) == len(l.note_indices) for l in lyrics.lines)


def test_an_unsingable_line_is_kept_and_flagged_not_swapped(melody):
    """A line with nothing to sing stays as written, unfitted, and the
    version says so - it is not quietly swapped for lexicon syllables."""
    slots = build_slots(melody)
    lines = ["..."] + [TAMIL_FIRST_LINE] * (len(slots) - 1)
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer(lines))
    assert lyrics.lines[0].text == "..."
    assert lyrics.lines[0].syllables == []
    assert "no singable syllables" in lyrics.notes.lower(), lyrics.notes
    assert lyrics.unfitted == 1
    assert lyrics.lines[1].text == TAMIL_FIRST_LINE


def test_lexicon_lines_say_so_and_swara_singing_is_still_a_line(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4)
    assert all(l.source == "lexicon" for l in lyrics.lines)
    slot = build_slots(melody)[0]
    syllables, _, warnings = fit_line("sa ri ga ma pa dha ni sa", slot)
    assert syllables and "".join(syllables).startswith("sa")


def test_a_lyric_line_s_source_survives_a_saved_project(tmp_path):
    from dataclasses import asdict
    from raagacomposer.core.models import LyricLine, Project
    from raagacomposer.core.persistence import ProjectStore
    from raagacomposer.core.settings import Settings
    settings = Settings()
    settings.projects_dir = str(tmp_path / "projects")
    store = ProjectStore(settings)
    project = Project(title="Provenance")
    from raagacomposer.core.models import LyricsVersion
    project.lyrics.append(LyricsVersion(lines=[
        LyricLine(text=TAMIL_FIRST_LINE, syllables=TAMIL_SLOTS,
                  note_indices=list(range(7)), source="creator")]))
    directory = store.ensure_dirs(tmp_path / "projects" / "prov")
    store.save(project, directory)
    again = store.open(directory)
    assert again.lyrics[0].lines[0].text == TAMIL_FIRST_LINE
    assert again.lyrics[0].lines[0].source == "creator"


def test_a_writer_s_empty_answer_for_a_requested_line_is_unfitted(melody):
    """An empty string is not words.  A blank answer for a slot that asked
    for words counts as unfitted, so the draft gate sees it (Arya's
    empty-lyric review)."""
    slots = build_slots(melody)
    lines = [""] + [TAMIL_FIRST_LINE] * (len(slots) - 1)
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer(lines))
    assert lyrics.lines[0].text == "" and lyrics.lines[0].syllables == []
    assert lyrics.lines[0].unfitted
    assert lyrics.unfitted == 1
    assert "nothing" in lyrics.notes.lower() or "empty" in lyrics.notes.lower()
    whitespace = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                          llm=_Writer(["   "] + [TAMIL_FIRST_LINE] * (len(slots) - 1)))
    assert whitespace.unfitted == 1


def test_the_unfitted_count_follows_every_refit(melody):
    """Invalid to valid, and valid back to invalid: the count is what the
    lines are now, not what they were when generated."""
    slots = build_slots(melody)
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer(["..."] + [TAMIL_FIRST_LINE] * (len(slots) - 1)))
    assert lyrics.unfitted == 1 and lyrics.lines[0].unfitted
    refit_line(lyrics, melody, lyrics.lines[0].id, TAMIL_FIRST_LINE)
    assert lyrics.lines[0].syllables and not lyrics.lines[0].unfitted
    assert lyrics.unfitted == 0
    refit_line(lyrics, melody, lyrics.lines[1].id, "...")
    assert lyrics.lines[1].unfitted and lyrics.unfitted == 1


# --------------------------------------------------------------------------
# a writer that answers with fewer lines than were asked for leaves the
# rest missing - never quietly filled with lexicon vocables (2026-09-08)
# --------------------------------------------------------------------------
def test_a_writer_s_missing_lines_stay_missing_not_lexicon(melody):
    slots = build_slots(melody)
    assert len(slots) > 2
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer([TAMIL_FIRST_LINE, "மலர்ந்தேன் நானே"]))
    assert [l.text for l in lyrics.lines[:2]] == [TAMIL_FIRST_LINE, "மலர்ந்தேன் நானே"]
    assert all(l.source == "llm:test-writer" for l in lyrics.lines[:2])
    rest = lyrics.lines[2:]
    assert all(l.text == "" and l.syllables == [] and l.unfitted for l in rest), \
        [(l.text, l.source) for l in rest]
    assert all(l.source == "missing:test-writer" for l in rest)
    assert lyrics.unfitted == len(slots) - 2
    assert not any(l.source == "lexicon" for l in lyrics.lines)
    for slot in slots[2:]:
        assert slot.section_name in lyrics.notes, (slot.section_name, lyrics.notes)


def test_a_writer_that_returns_nothing_leaves_every_requested_line_missing(melody):
    slots = build_slots(melody)
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4, llm=_Writer([]))
    assert lyrics.unfitted == len(slots)
    assert all(l.text == "" and l.source == "missing:test-writer" for l in lyrics.lines)
    # Without any writer at all, the lexicon is the writer and says so.
    plain = generate(melody, CreativeBrief(language="Tamil"), seed=4, llm=None)
    assert plain.unfitted == 0 and all(l.source == "lexicon" for l in plain.lines)


def test_missing_lines_respect_the_selected_sections_and_locks(melody):
    from raagacomposer.core.models import SectionKind
    slots = build_slots(melody)
    full = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                    llm=_Writer([TAMIL_FIRST_LINE] * len(slots)))
    full.lines[0].locked = True
    charanam = next(s for s in melody.sections if s.kind == SectionKind.CHARANAM)
    mine = [i for i, s in enumerate(slots) if s.section_id == charanam.id]
    assert len(mine) >= 2, "this tune's Charanam needs at least two phrases"
    again = generate(melody, CreativeBrief(language="Tamil"), seed=5,
                     llm=_Writer(["விடமாட்டேன்"]), previous=full,
                     section_ids=[charanam.id])
    assert again.lines[mine[0]].text == "விடமாட்டேன்"
    assert again.lines[mine[0]].source == "llm:test-writer"
    for i in mine[1:]:
        assert again.lines[i].text == "" and again.lines[i].unfitted
        assert again.lines[i].source == "missing:test-writer"
    assert again.unfitted == len(mine) - 1
    for i, line in enumerate(again.lines):
        if i not in mine:
            assert line.text == TAMIL_FIRST_LINE and line.source == "llm:test-writer"
            assert not line.unfitted
    assert again.lines[0].locked and again.lines[0].text == TAMIL_FIRST_LINE


def test_regenerate_line_with_a_silent_writer_leaves_the_line(melody):
    lyrics = generate(melody, CreativeBrief(language="Tamil"), seed=4,
                      llm=_Writer([TAMIL_FIRST_LINE] * len(build_slots(melody))))
    before = (lyrics.lines[0].text, list(lyrics.lines[0].syllables), lyrics.lines[0].source)
    warnings = regenerate_line(lyrics, melody, lyrics.lines[0].id,
                               CreativeBrief(language="Tamil"), seed=9, llm=_Writer([]))
    assert (lyrics.lines[0].text, list(lyrics.lines[0].syllables), lyrics.lines[0].source) == before
    assert warnings and "unchanged" in warnings[0].lower()
    assert lyrics.unfitted == 0
