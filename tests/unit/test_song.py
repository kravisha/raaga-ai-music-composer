"""Ravanan Thirumbittaan (studio/song.py): one syllable per note, in the raaga, sung over Ravana's Answer II."""
from __future__ import annotations

import pytest

from raagacomposer.studio import answer2 as A2
from raagacomposer.studio import song as S

pytestmark = pytest.mark.unit

SHANMUKHAPRIYA = {0, 2, 3, 6, 7, 8, 10}


def test_every_line_has_one_syllable_per_note_of_its_hook_bar():
    pairs = [(S.L1, A2.CALL[0]), (S.L2, A2.CALL[1]), (S.L3, A2.ANSWER[0]), (S.L4, A2.ANSWER[1]), (S.L4_UP, A2.ANSWER_UP),
             (S.C1, A2.CALL[0]), (S.C2, A2.CALL[1])]
    for line, bar in pairs:
        assert len(line.syllables) == len(bar), line.roman
        assert line.tamil and line.english


def test_the_vocal_line_is_in_the_raaga_and_follows_the_composition():
    sung, starts, seconds = S.vocal_line()
    _parts, _s, spans, _st = A2.build()
    assert sung and all(w for _n, w in sung)
    for n, _w in sung:
        tonic = next(t for a, b, t in spans if a - 1e-3 <= n.start < b - 1e-3)
        assert (n.midi - tonic) % 12 in SHANMUKHAPRIYA, (n.start, n.swara)
    first = min(n.start for n, _w in sung)
    assert first == pytest.approx(starts["answer"], abs=1e-3), "the singer opens his ten seconds with the pallavi"
    assert max(n.start + n.duration for n, _w in sung) <= seconds + 0.01


def test_every_sung_note_sits_on_a_band_note():
    """The singer doubles the veena or saxophone: same start, same pitch class (catches drift if the composition changes)."""
    sung, _starts, _seconds = S.vocal_line()
    parts, _s, _sp, _st = A2.build()
    band = [(round(n.start, 3), n.midi % 12) for k in ("veena", "saxophone") for n in parts[k]]
    lost = [(round(n.start, 2), n.swara) for n, _w in sung if (round(n.start, 3), n.midi % 12) not in band]
    assert not lost, lost[:10]


def test_the_lyric_sheet_has_tamil_roman_and_english():
    sheet = S.lyric_sheet()
    for line in (*S.PALLAVI, *S.CHARANAM):
        assert line.tamil in sheet and line.english in sheet


def test_it_renders_with_a_voice_over_the_band():
    audio = S.render(sr=16000, lead_in=1.0)
    assert audio.ndim == 2 and float(abs(audio[:8000]).max()) < 1e-3 and float(abs(audio).max()) > 0.1
