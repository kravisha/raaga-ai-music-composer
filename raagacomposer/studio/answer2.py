"""RAVANA'S ANSWER II - Claude Work's composition v5 (Krish, 7 Nov 2026).

Built on v4 (``answer.py``), the variation Krish liked.  The room's notes for the next take: "build from what worked, keep the
veena in the brief, but add a clear hook sooner and vary the rhythm so it doesn't feel locked in."

So:
* THE HOOK FIRST.  One bar of boos, then the veena plays the hook itself at 1.7 s (v4 held it back to 15 s).  His ten seconds
  are the hook, the saxophone's answer over stop-time hits, a tihai and sam.
* THE RHYTHM KEEPS MOVING.  Every section has its own feel, so the ear never settles:
    GROOVE A   straight four for four bars, then the veena's call comes a beat LATE (its tail landing on the next bar) over
               3-against-4 hats (dotted eighths), the answer back on the beat;
    HALF-TIME  the floor drops: kick on one, clap on three, the veena plays the hook at half speed, wide and heavy;
    KHANDA     a break in fives (5 + 5 + 6 sixteenths): mridangam and veena korvai in khanda nadai against the clap;
    GROOVE B   the whole-step lift from v4 (D -> E), double-time hats, call and answer tightened to one bar each;
    TIHAI      the ending from v4, with whistles.
Same raaga (Shanmukhapriya), same 140 bpm; every note from ``Raaga.midi``.
"""
from __future__ import annotations

import numpy as np

from ..audio import dsp
from ..core.models import Note
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import library
from .answer import (
    ANSWER,
    ANSWER_UP,
    BAR,
    CALL,
    FOUR,
    KICKS,
    KIT_CLAP,
    KIT_HAT,
    KIT_KICK,
    LIFT,
    MIX,
    RAAGA,
    ST,
    STEPS,
    TONIC,
    _Score,
)
from .groove import _step

SECTIONS = (("boos", 1), ("answer", 6), ("groove_a", 8), ("half_time", 4), ("khanda", 2), ("groove_b", 8), ("tihai", 2))
FEELS = {"groove_a": "straight, then displaced over 3:4", "half_time": "half-time", "khanda": "fives (5+5+6)",
         "groove_b": "double-time hats, lifted a whole step"}
KHANDA = (0, 5, 10)                       # where each group of five (and the last six) starts


def _phrase(sc: _Score, key: str, bar: tuple, tonic: int, t0: float, vel: int, *, octave: int = 0, late: int = 0,
            stretch: float = 1.0, up: int = 0) -> None:
    """A bar of the hook, optionally ``late`` by some sixteenths (the whole phrase moves later and its tail spills into the
    next bar's first beat - review: a rotation inside the bar turned the hook back to front) or stretched (half speed)."""
    for s, length, sw in bar:
        token = _step(sc.raaga, sw, up) if up else sw
        sc.note(key, token, tonic + 12 * octave, t0 + (s + late) * ST * stretch, length * ST * 0.92 * stretch, vel)


def build(seed: int = 5) -> tuple[dict[str, list[Note]], float, list[tuple[float, float, int]], dict[str, float]]:
    """-> (parts, seconds, [(start, end, tonic)], {section: start seconds})."""
    raaga = library().require(RAAGA)
    sc = _Score(raaga, seed)
    starts: dict[str, float] = {}
    t = 0.0
    for name, bars in SECTIONS:
        starts[name] = t
        t += bars * BAR
    total = t
    D, E = TONIC, TONIC + LIFT

    # BOOS: one bar now - the crowd, a heartbeat, prati Ma against Sa
    t0 = starts["boos"]
    for i, sw in enumerate(("P", "M2", "G2", "R2")):
        sc.note("choir", sw, D - 12, t0 + i * BAR / 4, BAR * 0.3, 124 - 6 * i)
        sc.note("choir", _step(raaga, sw, 2), D - 12, t0 + i * BAR / 4, BAR * 0.3, 112 - 6 * i)
    for s in (0, 3, 8, 11):
        sc.hit("drum_kit", t0 + s * ST, KIT_KICK, ST * 2, 118 if s in (0, 8) else 92)
    sc.note("strings", "S", D - 12, t0, BAR, 110)
    sc.note("strings", "M2", D - 12, t0, BAR, 100)

    # ANSWER: the hook first.  Bars 1-2 veena + claps, bars 3-4 sax over stop-time, bar 5 tihai, bar 6 sam and pick-up.
    t0 = starts["answer"]
    for b in range(2):
        _phrase(sc, "veena", CALL[b], D, t0 + b * BAR, 112)
        for s in (4, 12):
            sc.hit("drum_kit", t0 + b * BAR + s * ST, KIT_CLAP, ST, 96)
    sc.hit("drum_kit", t0, KIT_KICK, ST * 2, 110)
    for b in range(2):
        tb = t0 + (2 + b) * BAR
        _phrase(sc, "saxophone", ANSWER[b], D, tb, 110)
        for k, s in enumerate((0, 3, 6) if b == 0 else (0, 3, 6, 10, 12, 14)):
            chord = [_step(raaga, sw, k % 3) for sw in ("S", "G2", "P")]
            for sw in chord:
                sc.note("strings", sw, D, tb + s * ST, ST * 2, 104 + 4 * k)
            sc.hit("drum_kit", tb + s * ST, KIT_KICK, ST * 2, 118)
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 112)
    b5 = t0 + 4 * BAR
    sc.hit("mridangam", b5, 48, ST * 2, 110)
    sc.note("bass", "S", D - 24, b5, ST * 2, 112)
    for unit, start in enumerate((3, 8, 13)):
        for i, sw in enumerate(("P", "D1", "N2")):
            s = start + i
            sc.note("veena", sw, D + 12 * (unit == 2), b5 + s * ST, ST * 0.95, 96 + 8 * unit)
            sc.hit("mridangam", b5 + s * ST, 48 + i, ST, 92 + 10 * unit)
            sc.hit("drum_kit", b5 + s * ST, KIT_CLAP if i == 2 else KIT_HAT, ST * 0.6, 70 + 14 * unit)
    b6 = t0 + 5 * BAR
    for sw in ("S", "P", "S+"):
        sc.note("strings", sw, D, b6, ST * 4, 127)
    sc.note("veena", "S+", D + 12, b6, ST * 4, 124)
    sc.note("bass", "S", D - 24, b6, ST * 4, 127)
    for m in (KIT_KICK, KIT_CLAP, KIT_HAT):
        sc.hit("drum_kit", b6, m, ST * 6, 127)
    for i, (s, sw) in enumerate(((8, "N2"), (9, "D1"), (10, "P"), (12, "D1"), (13, "N2"), (14, "S+"))):
        sc.note("veena", sw, D, b6 + s * ST, ST * 0.95, 100 + 3 * i)
        sc.hit("drum_kit", b6 + s * ST, KIT_CLAP, ST * 0.6, 84 + 6 * i)
    sc.tonic_spans.append((0.0, starts["groove_b"], D))

    # GROOVE A: straight for four bars, then the veena's call a beat late over 3-against-4 hats (the answer stays on the beat)
    g0 = starts["groove_a"]
    last = None
    for bar in range(8):
        tb = g0 + bar * BAR
        displaced = bar >= 4
        kicks = KICKS[int(sc.rng.integers(len(KICKS)))]
        if kicks == last:
            kicks = KICKS[(KICKS.index(kicks) + 1) % len(KICKS)]
        last = kicks
        for s in kicks:
            sc.hit("drum_kit", tb + s * ST, KIT_KICK, ST * 2, 116 if s == 0 else 96)
        for s in (4, 12):
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 110)
        hats = range(0, STEPS, 3) if displaced else range(0, STEPS, 2)          # dotted eighths against the four
        for s in hats:
            sc.hit("drum_kit", tb + s * ST, KIT_HAT, ST * 0.4, 80 if s % 4 == 0 else 52)
        if bar % 4 == 3:
            for i, s in enumerate(range(12, 16)):
                sc.hit("mridangam", tb + s * ST, 48 + i, ST, 84 + 8 * i)
        sc.bass_bar(tb, bar, D, 100)
        k = bar % 4
        if k < 2:                       # the veena's call comes a beat LATE in the second half; its tail lands on the next bar
            _phrase(sc, "veena", CALL[k], D, tb, 106, late=4 if displaced else 0)
        else:                           # the answer stays on the beat, so the climb into the half-time drop lands
            _phrase(sc, "saxophone", ANSWER_UP if (k == 3 and displaced) else ANSWER[k - 2], D, tb, 104)
        sc.note("synth_pad", "S", D - 12, tb, BAR, 34)
        for s in (2, 6, 10, 14):
            sc.hit("tambourine", tb + s * ST, 61, ST * 0.6, 50)

    # HALF-TIME: kick on one, clap on three; the hook at half speed, two bars per hook bar
    h0 = starts["half_time"]
    for bar in range(4):
        tb = h0 + bar * BAR
        sc.hit("drum_kit", tb, KIT_KICK, ST * 4, 124)
        sc.hit("drum_kit", tb + 8 * ST, KIT_CLAP, ST * 2, 120)
        sc.hit("drum_kit", tb + 14 * ST, KIT_KICK, ST * 2, 80)
        for s in range(0, STEPS, 4):
            sc.hit("drum_kit", tb + s * ST, KIT_HAT, ST * 1.5, 60)
        sc.note("bass", ("S", "S", "D1", "P")[bar], D - 24, tb, BAR * 0.95, 116)
        sc.note("strings", ("S", "S", "D1", "P")[bar], D - 12, tb, BAR, 70)
    for half in range(2):                                     # CALL[0] over bars 1-2, CALL[1] over bars 3-4
        _phrase(sc, "veena", CALL[half], D, h0 + half * 2 * BAR, 110, stretch=2.0)
        _phrase(sc, "violin", CALL[half], D, h0 + half * 2 * BAR, 58, stretch=2.0, octave=1)

    # KHANDA: a break in fives - mridangam strokes 5 + 5 + 6 against the clap's four; a veena korvai in fives
    k0 = starts["khanda"]
    five = ("S+", "N2", "D1", "P", "M2")
    for bar in range(2):
        tb = k0 + bar * BAR
        for s in (4, 12):
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 100)
        for g, start in enumerate(KHANDA):
            length = 6 if g == 2 else 5
            for i in range(length):
                s = start + i
                sc.hit("mridangam" if bar == 0 else "tabla", tb + s * ST, 48 + (i % 4), ST, 110 if i == 0 else 64)
                if bar == 1 or g < 2:
                    sc.note("veena", five[i % 5] if i < 5 else "S+", D, tb + s * ST, ST * 0.9, 104 if i == 0 else 82)
    sc.note("bass", "S", D - 24, k0, BAR * 2, 96)
    for s in range(8, STEPS):
        if s == 12:                                   # the backbeat clap is already there (review: it was doubled)
            continue
        sc.hit("drum_kit", k0 + BAR + s * ST, KIT_CLAP, ST * 0.6, 50 + 9 * (s - 8))

    # GROOVE B: up a whole step; double-time hats; call and answer one bar each
    b0 = starts["groove_b"]
    for bar in range(8):
        tb = b0 + bar * BAR
        kicks = FOUR if bar % 2 == 0 else KICKS[int(sc.rng.integers(len(KICKS)))]   # FOUR is not in KICKS: never repeats
        for s in kicks:
            sc.hit("drum_kit", tb + s * ST, KIT_KICK, ST * 2, 118 if s == 0 else 100)
        for s in (4, 12):
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 114)
        for s in range(STEPS):                                                  # double-time hats
            sc.hit("drum_kit", tb + s * ST, KIT_HAT, ST * 0.3, 72 if s % 4 == 0 else (54 if s % 2 == 0 else 36))
        if bar % 4 == 3:
            for i, s in enumerate(range(10, 16)):
                sc.hit("mridangam", tb + s * ST, 48 + (i % 4), ST, 80 + 7 * i)
        else:
            for s in (0, 3, 6, 8, 11, 14):
                sc.hit("mridangam", tb + s * ST, 48 + (s % 4), ST, 76 if s in (0, 8) else 58)
        sc.bass_bar(tb, bar, E, 110)
        for s in (2, 6, 10, 14):
            sc.hit("tambourine", tb + s * ST, 61, ST * 0.6, 62)
        if bar % 2 == 0:                                                        # call - one bar
            _phrase(sc, "veena", CALL[(bar // 2) % 2], E, tb, 108, octave=1 if bar >= 4 else 0)
        else:                                                                   # answer - one bar
            _phrase(sc, "saxophone", ANSWER_UP if bar == 7 else ANSWER[(bar // 2) % 2], E, tb, 106)
            _phrase(sc, "violin", ANSWER_UP if bar == 7 else ANSWER[(bar // 2) % 2], E, tb, 68, octave=1)
        sc.note("choir", "S", E, tb, BAR, 40 + 5 * bar)
        sc.note("choir", "P", E, tb, BAR, 34 + 5 * bar)
        if bar >= 4:
            for _ in range(2):
                s = int(sc.rng.integers(0, STEPS - 2))
                sc.note("flute", "P+" if sc.rng.random() < 0.5 else "S+", E + 12, tb + s * ST, ST * 1.5, 60)

    # TIHAI: as v4
    t0 = starts["tihai"]
    for u, start in enumerate((0, 6, 12)):
        for d, sw in enumerate(("S+", "N2", "D1", "P")):
            tt = t0 + (start + d) * ST
            for key, vel in (("veena", 106), ("saxophone", 104), ("violin", 80)):
                sc.note(key, sw, E + (12 if key == "violin" else 0), tt, ST * 0.95, vel + 6 * u)
            sc.hit("drum_kit", tt, KIT_KICK if d == 0 else KIT_CLAP, ST, 104 + 6 * u)
            sc.hit("mridangam", tt, 48 + d, ST, 96 + 8 * u)
    end = t0 + BAR
    for sw in ("S", "P", "S+"):
        sc.note("strings", sw, E, end, BAR * 0.9, 127)
    sc.note("veena", "S+", E, end, BAR * 0.9, 127)
    sc.note("saxophone", "S", E, end, BAR * 0.9, 120)
    sc.note("bass", "S", E - 24, end, BAR * 0.9, 127)
    sc.note("choir", "S+", E, end, BAR, 80)
    for m in (KIT_KICK, KIT_CLAP, KIT_HAT):
        sc.hit("drum_kit", end, m, ST * 8, 127)
    for i in range(6):
        sc.note("flute", ("S+", "P+", "S+", "P+", "S++", "S+")[i], E + 12, end + (2 + i * 2) * ST, ST * 1.6, 70 - 4 * i)
    sc.tonic_spans.append((starts["groove_b"], total + 2.0, E))
    return sc.parts, total, sc.tonic_spans, starts


def render(sr: int = 44100, seed: int = 5, lead_in: float = 0.0) -> np.ndarray:
    parts, seconds, _spans, _starts = build(seed)
    total = lead_in + seconds + 2.5
    mix = dsp.silence(total, sr)
    for k, (key, notes) in enumerate(sorted(parts.items())):
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        shifted = [Note(swara=n.swara, midi=n.midi, start=round(n.start + lead_in, 4), duration=n.duration, velocity=n.velocity)
                   for n in notes]
        gain, pan = MIX.get(key, (0.6, 0.0))
        mix = dsp.mix_into(mix, dsp.pan_mono(render_notes(shifted, inst, sr, total_seconds=total, seed=seed + k) * gain, pan), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.5, sr).astype(np.float32)
