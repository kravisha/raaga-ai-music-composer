"""Selected swara passages must preserve the rest of a song and render."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from raagacomposer.core.models import (
    ApprovalState, LyricLine, LyricsVersion, MelodyVersion, Note, Section,
    SectionKind, VocalDirection,
)
from raagacomposer.core.versioning import LockedContentError
from raagacomposer.voice.profiles import BUILTIN
from raagacomposer.voice.renderer import plan_segments, render
from raagacomposer.voice.swara import prepare_swara_singing, swara_syllable

pytestmark = pytest.mark.unit


@pytest.fixture
def song(keeravani):
    sections = [
        Section(id="pre", name="Prelude", kind=SectionKind.PRELUDE,
                start=0, end=1),
        Section(id="pal", name="Pallavi", kind=SectionKind.PALLAVI,
                start=1, end=3),
        Section(id="inter", name="Interlude", kind=SectionKind.INTERLUDE,
                start=3, end=4),
        Section(id="char", name="Charanam", kind=SectionKind.CHARANAM,
                start=4, end=5.5),
    ]
    events = [
        ("pre", "P-", 0),
        ("pal", "R2", 1), ("pal", "G2", 1.5), ("pal", "S+", 2.3),
        ("inter", "D1", 3), ("inter", "N3", 3.5),
        ("char", "M1", 4), ("char", "S", 4.8),
    ]
    melody = MelodyVersion(
        version=3, raaga=keeravani.name, tempo_bpm=120, sections=sections,
        notes=[Note(swara=swara, midi=keeravani.midi(swara, 60),
                    section_id=sid, start=start, duration=0.5,
                    velocity=77, gamaka="kampita" if swara == "G2" else "")
               for sid, swara, start in events],
        state=ApprovalState.LOCKED,
    )
    lyrics = LyricsVersion(
        version=4, melody_version=3, language="Tamil",
        state=ApprovalState.APPROVED, notes="Keep the accepted tune.",
        lines=[
            LyricLine(id="pal-line", section_id="pal", text="nee vaa nee",
                      syllables=["nee", "vaa", "nee"], note_indices=[1, 2, 3],
                      start=1, end=2.8),
            LyricLine(id="char-line", section_id="char", text="vaa nee",
                      syllables=["vaa", "nee"], note_indices=[6, 7],
                      start=4, end=5.3, locked=True),
        ],
    )
    return melody, lyrics


@pytest.mark.parametrize("token, syllable", [
    ("S+", "sa"), ("R2", "ri"), ("G1", "ga"), ("M2", "ma"),
    ("P-", "pa"), ("D3++", "dha"), ("N3", "ni"),
])
def test_names_follow_swara_spelling_not_enharmonic_pitch(token, syllable):
    assert swara_syllable(token) == syllable


def test_selected_passage_keeps_pitch_rhythm_ornaments_and_other_lyrics(song, keeravani):
    melody, lyrics = song
    original = (asdict(melody), asdict(lyrics))
    result = prepare_swara_singing(melody, keeravani, section_ids=["pal"], lyrics=lyrics)

    assert (asdict(melody), asdict(lyrics)) == original
    assert result.lyrics.version == 5
    assert result.lyrics.state == ApprovalState.DRAFT
    assert result.lyrics.language == "Tamil"
    assert result.lyrics.melody_version == 3
    assert result.lyrics.notes.startswith(lyrics.notes)
    assert result.section_ids == ("pal",)
    assert [s.syllable for s in result.segments] == ["ri", "ga", "sa", "vaa", "nee"]
    sung_notes = [melody.notes[i] for i in [1, 2, 3, 6, 7]]
    for segment, note in zip(result.segments, sung_notes):
        assert (segment.midi, segment.start, segment.end, segment.velocity, segment.gamaka) == (
            note.midi, note.start, note.end, note.velocity, note.gamaka)
    kept = result.lyrics.line_by_id("char-line")
    assert asdict(kept) == asdict(lyrics.lines[1])
    kept.syllables[0] = "changed only in the draft"
    assert lyrics.lines[1].syllables == ["vaa", "nee"]
    # The rest before the upper tonic still separates the breath phrases.
    assert [l.note_indices for l in result.lyrics.lines if l.section_id == "pal"] == [[1, 2], [3]]


def test_only_requested_instrumental_section_gets_vocals(song, keeravani):
    melody, lyrics = song
    result = prepare_swara_singing(melody, keeravani,
                                   section_ids=["inter", "pal"], lyrics=lyrics)
    assert result.section_ids == ("pal", "inter")
    assert [s.syllable for s in result.segments] == ["ri", "ga", "sa", "dha", "ni", "vaa", "nee"]
    assert result.segments[0].start == 1  # Prelude remains instrumental.
    assert result.segments[5].legato  # Charanam follows the sung interlude.


def test_unsung_interlude_does_not_create_false_vocal_legato(song, keeravani):
    melody, lyrics = song
    result = prepare_swara_singing(melody, keeravani, section_ids=["pal"], lyrics=lyrics)
    charanam = next(s for s in result.segments if s.start == 4)
    assert not charanam.legato


def test_no_selection_preserves_existing_approved_lyrics(song, keeravani):
    melody, lyrics = song
    result = prepare_swara_singing(melody, keeravani, section_ids=[], lyrics=lyrics)
    assert result.lyrics is not lyrics
    assert asdict(result.lyrics) == asdict(lyrics)
    assert result.segments == plan_segments(melody, lyrics)
    assert result.section_ids == ()


def test_swaras_can_be_added_before_lyric_generation(song, keeravani):
    melody, _ = song
    result = prepare_swara_singing(melody, keeravani, section_ids=["inter"])
    assert result.lyrics.version == 1
    assert result.lyrics.melody_version == melody.version
    assert [l.text for l in result.lyrics.lines] == ["dha ni"]
    assert [s.syllable for s in result.segments if 3 <= s.start < 4] == ["dha", "ni"]


@pytest.mark.parametrize("locked", ["lyrics", "section", "line"])
def test_locked_affected_content_is_rejected_without_mutation(song, keeravani, locked):
    melody, lyrics = song
    if locked == "lyrics":
        lyrics.state = ApprovalState.LOCKED
    elif locked == "section":
        melody.sections[1].locked = True
    else:
        lyrics.lines[0].locked = True
    original = (asdict(melody), asdict(lyrics))
    with pytest.raises(LockedContentError):
        prepare_swara_singing(melody, keeravani, section_ids=["pal"], lyrics=lyrics)
    assert (asdict(melody), asdict(lyrics)) == original


def test_stale_lyric_alignment_is_rejected(song, keeravani):
    melody, lyrics = song
    lyrics.melody_version = 2
    with pytest.raises(ValueError, match="different melody"):
        prepare_swara_singing(melody, keeravani, section_ids=["pal"], lyrics=lyrics)


@pytest.mark.parametrize("invalid", ["missing-section", "empty-section", "bad-token",
                                    "wrong-raaga", "foreign-swara", "wrong-pitch",
                                    "zero-duration", "outside-section", "nan-time"])
def test_invalid_passages_fail_clearly(song, keeravani, invalid):
    melody, lyrics = song
    selected = ["pal"]
    if invalid == "missing-section":
        selected = ["removed"]
    elif invalid == "empty-section":
        melody.notes = [n for n in melody.notes if n.section_id != "pal"]
    elif invalid == "bad-token":
        melody.notes[1].swara = "Sa"
    elif invalid == "wrong-raaga":
        melody.raaga = "Mohanam"
    elif invalid == "foreign-swara":
        melody.notes[1].swara = "M2"
    elif invalid == "wrong-pitch":
        melody.notes[1].midi += 12
    elif invalid == "zero-duration":
        melody.notes[1].duration = 0
    elif invalid == "outside-section":
        melody.notes[1].start = 0.5
    else:
        melody.notes[1].start = float("nan")
    with pytest.raises(ValueError):
        prepare_swara_singing(melody, keeravani, section_ids=selected, lyrics=lyrics)


@pytest.mark.parametrize("selected_line", [True, False])
def test_alignment_cannot_overwrite_another_sections_notes(song, keeravani, selected_line):
    melody, lyrics = song
    if selected_line:
        lyrics.lines[0].note_indices.append(6)
    else:
        lyrics.lines[1].note_indices.append(1)
    with pytest.raises(ValueError, match="crosses"):
        prepare_swara_singing(melody, keeravani, section_ids=["pal"], lyrics=lyrics)


def test_swara_segments_render_audible_syllables_and_leave_prelude_silent(song, keeravani):
    melody, lyrics = song
    result = prepare_swara_singing(melody, keeravani,
                                   section_ids=["pal", "inter"], lyrics=lyrics)
    sr = 16000
    audio = render(result.segments, BUILTIN[2], VocalDirection(),
                   sr=sr, total_seconds=6, seed=19)
    assert audio.shape == (6 * sr,)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio[int(1.1 * sr):int(3.9 * sr)])) > 0.01
    assert np.max(np.abs(audio[:int(0.8 * sr)])) < 1e-4
    assert [s.consonant for s in result.segments[:5]] == ["r", "g", "s", "dh", "n"]
    assert [s.vowel for s in result.segments[:5]] == ["i", "a", "a", "a", "i"]
    open_vowels = deepcopy(result.segments)
    for segment in open_vowels:
        segment.consonant, segment.vowel = "", "a"
    plain = render(open_vowels, BUILTIN[2], VocalDirection(),
                   sr=sr, total_seconds=6, seed=19)
    assert not np.allclose(audio, plain)
