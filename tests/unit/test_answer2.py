"""Ravana's Answer II (studio/answer2.py): the hook sooner, the rhythm never locked in, still in the raaga."""
from __future__ import annotations

import pytest

from raagacomposer.music import instruments as catalog
from raagacomposer.studio import answer2 as A2

pytestmark = pytest.mark.unit

SHANMUKHAPRIYA = {0, 2, 3, 6, 7, 8, 10}
PITCHED = {k for k in A2.MIX if not getattr(catalog.get(k), "percussive", True)}


def _tonic_at(spans, t):
    for start, end, tonic in spans:
        if start - 1e-3 <= t < end - 1e-3:
            return tonic
    raise AssertionError(f"no section at {t:.2f}s")


def test_every_pitch_is_in_shanmukhapriya_at_its_sections_tonic():
    parts, _s, spans, _st = A2.build()
    bad = [(k, round(n.start, 2), n.swara) for k in PITCHED & set(parts) for n in parts[k]
           if (n.midi - _tonic_at(spans, n.start)) % 12 not in SHANMUKHAPRIYA]
    assert not bad, bad[:10]


def test_the_hook_comes_in_under_two_seconds():
    parts, _s, _sp, starts = A2.build()
    first_veena = min(n.start for n in parts["veena"])
    assert first_veena == pytest.approx(starts["answer"], abs=1e-3) and first_veena < 2.0


def _bar_signature(parts, t0):
    def steps(pred):
        return tuple(sorted({round((n.start - t0) / A2.ST) for n in parts["drum_kit"] if pred(n) and t0 - 1e-3 <= n.start < t0 + A2.BAR - 1e-3}))
    return (steps(lambda n: n.midi == A2.KIT_KICK), steps(lambda n: n.midi == A2.KIT_HAT))


def test_each_section_has_its_own_feel():
    parts, _s, _sp, starts = A2.build()
    feels = {name: _bar_signature(parts, starts[name] + (4 * A2.BAR if name == "groove_a" else 0))
             for name in ("groove_a", "half_time", "groove_b")}
    feels["groove_a_straight"] = _bar_signature(parts, starts["groove_a"])
    assert len(set(feels.values())) == 4, feels
    khanda = [n for n in parts["mridangam"] if starts["khanda"] - 1e-3 <= n.start < starts["khanda"] + A2.BAR - 1e-3]
    accents = sorted(round((n.start - starts["khanda"]) / A2.ST) for n in khanda if n.velocity >= 110)
    assert accents == [0, 5, 10], "groups of five"


def test_no_two_groove_bars_in_a_row_share_a_kick_pattern():
    parts, _s, _sp, starts = A2.build()
    for name in ("groove_a", "groove_b"):
        sigs = [_bar_signature(parts, starts[name] + b * A2.BAR)[0] for b in range(8)]
        assert all(sigs[i] != sigs[i + 1] for i in range(7)), (name, sigs)


def test_the_late_call_keeps_the_hook_in_order_and_one_beat_later():
    parts, _s, _sp, starts = A2.build()
    for bar, hook in ((4, A2.CALL[0]), (5, A2.CALL[1])):
        t0 = starts["groove_a"] + bar * A2.BAR
        got = [n for n in parts["veena"] if t0 + 4 * A2.ST - 1e-3 <= n.start < t0 + 20 * A2.ST - 1e-3 and n.velocity == 106]
        got = sorted(got, key=lambda n: n.start)[:len(hook)]
        assert [n.swara for n in got] == [sw for _s, _l, sw in hook], "same notes, same order"
        assert round((got[0].start - t0) / A2.ST) == 4, "a beat late"


def test_same_seed_same_piece():
    a, b = A2.build(5)[0], A2.build(5)[0]
    assert {k: [(n.start, n.midi) for n in v] for k, v in a.items()} == {k: [(n.start, n.midi) for n in v] for k, v in b.items()}


def test_v6_moves_the_answer_to_the_flute_and_the_strokes_to_the_hand_drum():
    parts, _s, _sp, _st = A2.build()
    for hand in ("tabla", "bongos"):
        out = A2.orchestrate(parts, answer="flute", hand=hand)
        assert "saxophone" not in out and "mridangam" not in out
        assert out["flute"] == parts["flute"], "the whistles keep their own part and mix (review)"
        assert len(out["flute@saxophone"]) == len(parts["saxophone"])
        moved = out[f"{hand}@mridangam"]
        assert len(moved) == len(parts["mridangam"])
        inst = catalog.get(hand)
        assert inst is not None
        strokes = len(inst.hit_freqs)
        assert all(0 <= n.midi - 48 < strokes for n in moved)
        top = [m.midi for n, m in zip(parts["mridangam"], moved) if n.midi == 51]
        assert all(t == 48 + strokes - 1 for t in top), "a climbing fill ends on the top drum, never back on the lowest"
    assert A2.orchestrate(parts) == parts, "the defaults leave v5 as it was"


def test_krishs_pick_is_veena_flute_and_bongos():
    assert A2.CHOSEN["hand"] == "bongos" and A2.CHOSEN["answer"] == "flute" and A2.CHOSEN["veena"] > 1.0


def test_v6_renders_with_tabla_and_with_bongos():
    for hand in ("tabla", "bongos"):
        audio = A2.render(sr=16000, **{**A2.V6, "hand": hand})
        assert audio.ndim == 2 and float(abs(audio).max()) > 0.1


def test_it_renders():
    audio = A2.render(sr=16000, lead_in=1.0)
    assert audio.ndim == 2 and float(abs(audio[:8000]).max()) < 1e-3 and float(abs(audio).max()) > 0.1
