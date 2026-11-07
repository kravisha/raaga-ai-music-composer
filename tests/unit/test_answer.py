"""Ravana's Answer (studio/answer.py): in the raaga, not monotonous, the right shape."""
from __future__ import annotations

import pytest

from raagacomposer.music import instruments as catalog
from raagacomposer.studio import answer as A

pytestmark = pytest.mark.unit

SHANMUKHAPRIYA = {0, 2, 3, 6, 7, 8, 10}          # S R2 G2 M2 P D1 N2
PITCHED = {k for k in A.MIX if not getattr(catalog.get(k), "percussive", True)}       # every pitched part, so a new one cannot slip by (review)
assert PITCHED >= {"veena", "saxophone", "violin", "strings", "choir", "flute", "bass", "synth_pad"}


def _tonic_at(spans, t):
    for start, end, tonic in spans:
        if start - 1e-3 <= t < end - 1e-3:            # note times are rounded to 0.1 ms
            return tonic
    raise AssertionError(f"no section at {t:.2f}s")


def test_every_sounding_pitch_is_in_shanmukhapriya_at_its_sections_tonic():
    parts, _seconds, spans = A.build()
    bad = []
    for key in PITCHED & set(parts):
        for n in parts[key]:
            pc = (n.midi - _tonic_at(spans, n.start)) % 12
            if pc not in SHANMUKHAPRIYA:
                bad.append((key, round(n.start, 2), n.swara, n.midi))
    assert not bad, bad[:10]


def test_the_twist_lifts_the_second_groove_a_whole_step():
    _parts, _s, spans = A.build()
    tonics = [t for _a, _b, t in spans]
    assert tonics[0] == A.TONIC and tonics[-1] == A.TONIC + A.LIFT


def test_the_answer_section_is_his_ten_seconds_and_the_piece_is_short_and_full():
    bars = dict(A.SECTIONS)
    assert 9.5 <= bars["answer"] * A.BAR <= 11.0
    _parts, seconds, _ = A.build()
    assert 40 <= seconds <= 60


def test_the_drums_never_play_the_same_bar_twice_in_a_row():
    parts, _s, _ = A.build()
    start = sum(b for name, b in A.SECTIONS[:2]) * A.BAR
    kicks = [n for n in parts["drum_kit"] if n.midi == A.KIT_KICK]
    for groove_from in (start, start + (8 + 2) * A.BAR):
        bars = []
        for b in range(8):
            t0 = groove_from + b * A.BAR
            bars.append(tuple(sorted(round((n.start - t0) / A.ST) for n in kicks if t0 - 1e-6 <= n.start < t0 + A.BAR - 1e-6)))
        assert all(bars[i] != bars[i + 1] for i in range(7)), bars


def test_the_veena_leads():
    parts, _s, _ = A.build()
    counts = {k: len(v) for k, v in parts.items() if k in PITCHED}
    assert max(counts, key=lambda k: counts[k]) in ("veena", "bass")
    assert counts["veena"] > counts["saxophone"]


def test_it_renders():
    audio = A.render(sr=16000, lead_in=1.0)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert float(abs(audio[:16000 // 2]).max()) < 1e-3, "the lead-in is quiet"
    assert float(abs(audio).max()) > 0.1
