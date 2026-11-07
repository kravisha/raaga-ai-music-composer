"""The full song (studio/full_song.py): a start, words through it, the ending; in the raaga; the voice on the band."""
from __future__ import annotations

import pytest

from raagacomposer.music import instruments as catalog
from raagacomposer.studio import full_song as F

pytestmark = pytest.mark.unit

SHANMUKHAPRIYA = {0, 2, 3, 6, 7, 8, 10}


def _tonic(spans, t):
    for a, b, tonic in spans:
        if a - 1e-3 <= t < b - 1e-3:
            return tonic
    raise AssertionError(t)


def test_every_pitch_including_the_voice_is_in_the_raaga():
    band, sung, _s, spans, _st = F.build()
    pitched = {k for k in band if not getattr(catalog.get("flute" if k == "whistle" else k), "percussive", True)}
    notes = [(k, n) for k in pitched for n in band[k]] + [("voice", n) for n, _w in sung]
    bad = [(k, round(n.start, 2), n.swara) for k, n in notes if (n.midi - _tonic(spans, n.start)) % 12 not in SHANMUKHAPRIYA]
    assert not bad, bad[:10]


def test_it_has_a_start_words_and_an_end():
    band, sung, seconds, _sp, starts = F.build()
    assert F.SECTIONS[0][0] == "intro" and F.SECTIONS[-1][0] == "ending"
    first_word = min(n.start for n, _w in sung)
    assert first_word == pytest.approx(starts["pallavi"], abs=1e-3), "the singer comes in with the pallavi, after the intro"
    assert min(n.start for n in band["veena"]) < 2.0, "the hook is heard in the first two seconds"
    sections_sung = {name for name, _b in F.SECTIONS
                     if any(starts[name] - 1e-3 <= n.start < starts[name] + _b * F.BAR - 1e-3 for n, _w in sung)}
    assert sections_sung == {"pallavi", "charanam", "pallavi_up", "ending"}
    assert 55 <= seconds <= 75


def test_every_line_fits_and_the_new_charanam_lines_lean_soft():
    for line, bar in ((F.C3, F.ANSWER[0]), (F.C4, F.ANSWER[1])):
        assert len(line.syllables) == len(bar)
    soft, hard = set("ஙஞணநமன"), set("கசடதபற")
    for line in (F.C3, F.C4):
        assert sum(c in soft for c in line.tamil) >= sum(c in hard for c in line.tamil), line.tamil


def test_the_voice_sings_on_the_bands_notes():
    band, sung, _s, _sp, _st = F.build()
    played = {(round(n.start, 3), n.midi % 12) for k in ("veena", F.ANSWER_INST) for n in band[k]}
    lost = [(round(n.start, 2), n.swara) for n, _w in sung if (round(n.start, 3), n.midi % 12) not in played]
    assert not lost, lost[:10]


def test_the_ending_is_the_one_krish_liked():
    _band, sung, _s, _sp, starts = F.build()
    words =[w for n, w in sung if n.start >= starts["ending"] - 1e-3]
    assert words == ["raa", "va", "nan", "daa"] * 3 + ["aa"]


def test_drum_strokes_are_real_hits_and_the_words_are_in_time_order():
    band, sung, _s, _sp, _st = F.build()
    for key in (F.HAND, "tabla"):
        inst = catalog.get(key)
        assert inst is not None
        assert all(0 <= n.midi - 48 < len(inst.hit_freqs) for n in band[key]), key
    times = [n.start for n, _w in sung]
    assert times == sorted(times)


def test_the_interlude_never_puts_the_veena_on_the_flutes_answer():
    band, _sung, _s, _sp, starts = F.build()
    answer_from = starts["interlude"] + 2 * F.BAR
    clash = [n for n in band["veena"] if answer_from - 1e-3 <= n.start < answer_from + 2 * F.BAR - 1e-3]
    assert not clash, [(round(n.start, 2), n.swara) for n in clash]


def test_the_voice_is_set_against_the_band_on_purpose():
    import numpy as np
    band = np.ones((1000, 2), dtype=np.float32) * 0.1
    vocal = np.zeros((1000, 2), dtype=np.float32)
    vocal[500:] = 0.5
    g = F.vocal_balance(band, vocal)
    assert 0.5 * g == pytest.approx(0.1 * 0.8 * 10 ** (F.VOICE_OVER_BAND_DB / 20), rel=1e-3)


def test_it_renders():
    audio = F.render(sr=16000, lead_in=1.0)
    assert audio.ndim == 2 and float(abs(audio[:8000]).max()) < 1e-3 and float(abs(audio).max()) > 0.1
