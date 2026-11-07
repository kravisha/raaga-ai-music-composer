"""RAVANA'S ANSWER - Claude Work's composition v4 for the Agentic Band (Krish, 7 Nov 2026).

Krish: "I like Claude Work's composition a lot ... give me some more of your composition."  On the other take: "too slow, too
predictable, too monotonous. It should be catchy, unpredictable, and not monotonous."  The scene: a young artist the crowd has
given up on - they boo him - and he has ten seconds to prove them wrong.  "Ravana loves veena. Let's go for the veena."

Same world as the piece Krish liked (``groove.py``): Shanmukhapriya in D at 140 bpm, four beats of sixteen steps; the inspiration
clip's feel, none of its notes.  New here, to keep the ear guessing:

  BOOS      2 bars  a low choir moaning down the raaga over a heartbeat kick; Sa against prati Ma (M2) keeps it uneasy.
  ANSWER    6 bars  (10.3 s - his ten seconds) veena alone; a beat of dead air; a 16th run; a stab that lands two steps LATE;
                    tresillo stop-time; a mridangam-and-veena tihai (3 x 3 strokes) that lands on sam with everything.
  GROOVE A  8 bars  a new syncopated hook: veena calls, saxophone answers; the drums never play the same bar twice in a row
                    (kick patterns rotate, ghost snares move, open hats, a fill every fourth bar).
  BREAK     2 bars  the kick drops out; claps, mridangam and tabla trade a bar each, the veena runs a korvai; a snare ramp.
  GROOVE B  8 bars  THE TWIST: everything jumps a whole step (D -> E, same raaga), four on the floor every other bar (the rest
                    rotate as in A), 16th hats, mridangam joins, violin
                    an octave over the answer, the crowd's "aa" rising underneath and whistles in the last four bars.
  TIHAI     2 bars  the last phrase three times with every instrument, a final hit on Sa, whistles over the ring-out.

Every note is computed from the raaga (``Raaga.midi``), so nothing leaves Shanmukhapriya at its section's tonic.
"""
from __future__ import annotations

import numpy as np

from ..audio import dsp
from ..core.models import Note
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import Raaga, library
from .groove import _step

BPM = 140
STEPS = 16
TONIC = 62                     # D
LIFT = 2                       # groove B: up a whole step, to E
RAAGA = "Shanmukhapriya"
ST = 60.0 / BPM / 4            # one sixteenth
BAR = STEPS * ST

SECTIONS = (("boos", 2), ("answer", 6), ("groove_a", 8), ("break", 2), ("groove_b", 8), ("tihai", 2))

#: the new hook: (step, length, swara).  Bars 1-2 the veena's call, bars 3-4 the saxophone's answer.  3+3+2 accents.
CALL = (
    ((0, 2, "S+"), (3, 1, "N2"), (4, 1, "D1"), (6, 2, "P"), (9, 1, "M2"), (10, 2, "P"), (13, 3, "D1")),
    ((0, 1, "N2"), (1, 1, "D1"), (2, 2, "P"), (4, 1, "M2"), (5, 1, "G2"), (6, 3, "M2"), (10, 2, "P"), (12, 1, "D1"),
     (13, 1, "N2"), (14, 2, "S+")),
)
ANSWER = (
    ((0, 3, "R2+"), (3, 3, "S+"), (6, 2, "N2"), (8, 1, "D1"), (9, 1, "P"), (10, 2, "D1"), (12, 4, "P")),
    ((0, 1, "M2"), (1, 1, "P"), (2, 1, "M2"), (3, 1, "G2"), (4, 2, "R2"), (6, 2, "G2"), (8, 6, "S")),
)
#: the second time round the answer does not come home - it flies up instead
ANSWER_UP = ((0, 1, "M2"), (1, 1, "P"), (2, 1, "M2"), (3, 1, "G2"), (4, 2, "R2"), (6, 2, "G2"), (8, 2, "P"), (10, 2, "N2"),
             (12, 4, "S+"))

BASS_ROOTS = ("S", "S", "D1-", "P-")
BASS_STEPS = ((0, 0), (3, 0), (6, 12), (10, 0), (12, 7), (14, 12))      # (step, semitone class: 0 root, 12 octave, 7 its fifth)

#: kick patterns the groove rotates through, so no two bars in a row are the same
KICKS = ((0, 6, 10), (0, 3, 8, 11), (0, 8, 9, 14), (0, 3, 6, 10, 13), (0, 7, 10))
FOUR = (0, 4, 8, 12)

KIT_KICK, KIT_CLAP, KIT_HAT = 36, 37, 38          # drum kit: midi % 3 -> 0 kick, 1 clap, 2 hat


class _Score:
    def __init__(self, raaga: Raaga, seed: int) -> None:
        self.raaga = raaga
        self.parts: dict[str, list[Note]] = {}
        self.rng = np.random.default_rng(seed)
        self.tonic_spans: list[tuple[float, float, int]] = []

    def hit(self, key: str, t: float, midi: int, dur: float, vel: int) -> None:
        self.parts.setdefault(key, []).append(Note(swara="-", midi=midi, start=round(t, 4), duration=round(dur, 4),
                                                   velocity=int(max(1, min(127, vel)))))

    def note(self, key: str, sw: str, tonic: int, t: float, dur: float, vel: int) -> None:
        low = len(sw) - len(sw.rstrip("-"))
        token = sw.rstrip("-")
        midi = self.raaga.midi(token, tonic) - 12 * low
        self.parts.setdefault(key, []).append(Note(swara=sw, midi=midi, start=round(t, 4), duration=round(dur, 4),
                                                   velocity=int(max(1, min(127, vel)))))

    def phrase(self, key: str, bar: tuple, tonic: int, t0: float, vel: int, octave: int = 0, late: int = 0,
               up: int = 0) -> None:
        for s, length, sw in bar:
            if s + late >= STEPS:
                continue
            token = _step(self.raaga, sw.rstrip("-"), up) if up else sw
            self.note(key, token, tonic + 12 * octave, t0 + (s + late) * ST, length * ST * 0.92, vel)

    # -- drums ---------------------------------------------------------------
    def kit_bar(self, t0: float, kicks, *, fill: bool, open_hat: bool, intensity: float) -> None:
        for s in kicks:
            self.hit("drum_kit", t0 + s * ST, KIT_KICK, ST * 2, 116 if s == 0 else 96)
        for s in (4, 12):
            self.hit("drum_kit", t0 + s * ST, KIT_CLAP, ST, 112)
        ghosts = [s for s in (2, 7, 9, 11, 15) if self.rng.random() < 0.45]
        for s in ghosts:
            self.hit("drum_kit", t0 + s * ST, KIT_CLAP, ST * 0.6, 40)
        for s in range(0, STEPS, 1 if intensity > 0.7 else 2):
            self.hit("drum_kit", t0 + s * ST, KIT_HAT, ST * (1.6 if open_hat and s == 14 else 0.4), 78 if s % 4 == 0 else 46)
        if fill:
            for i, s in enumerate(range(12, 16)):
                self.hit("drum_kit", t0 + s * ST, KIT_CLAP, ST * 0.7, 80 + 12 * i)
                self.hit("mridangam", t0 + s * ST, 48 + (i % 4), ST, 84 + 8 * i)

    def bass_bar(self, t0: float, bar: int, tonic: int, vel: int = 100) -> None:
        root = BASS_ROOTS[bar % len(BASS_ROOTS)]
        base = self.raaga.midi(root.rstrip("-"), tonic - 24) - 12 * (len(root) - len(root.rstrip("-")))
        for s, semis in BASS_STEPS:
            midi = base + semis
            if semis == 7:                              # the fifth of the root, taken along the raaga (P over S, S over P...)
                sw = _step(self.raaga, root.rstrip("-"), 4)
                midi = self.raaga.midi(sw.rstrip("+"), tonic - 24) + 12 * (len(sw) - len(sw.rstrip("+")))
                midi -= 12 * (len(root) - len(root.rstrip("-")))
            self.parts.setdefault("bass", []).append(Note(swara="-", midi=midi, start=round(t0 + s * ST, 4),
                                                          duration=round(ST * 1.7, 4), velocity=vel if s % 8 == 0 else vel - 16))


def build(seed: int = 4) -> tuple[dict[str, list[Note]], float, list[tuple[float, float, int]]]:
    """-> (parts, seconds, [(start, end, tonic)] for each stretch of the piece)."""
    raaga = library().require(RAAGA)
    sc = _Score(raaga, seed)
    starts: dict[str, float] = {}
    t = 0.0
    for name, bars in SECTIONS:
        starts[name] = t
        t += bars * BAR
    total = t
    D, E = TONIC, TONIC + LIFT

    # BOOS: the crowd moans down the raaga; a heartbeat kick; a low tanpura-like drone on Sa and prati Ma
    t0 = starts["boos"]
    for i, sw in enumerate(("P", "M2", "G2", "R2")):
        sc.note("choir", sw, D - 12, t0 + i * BAR / 2, BAR * 0.55, 124 - 6 * i)        # loud enough to be a crowd (first
        sc.note("choir", _step(raaga, sw, 2), D - 12, t0 + i * BAR / 2, BAR * 0.55, 112 - 6 * i)   # render sat 19 dB under)
        sc.note("choir", _step(raaga, sw, 4), D, t0 + i * BAR / 2, BAR * 0.5, 96 - 6 * i)
    for b in range(2):
        for s in (0, 3):
            sc.hit("drum_kit", t0 + b * BAR + s * ST, KIT_KICK, ST * 2, 118 if s == 0 else 96)
    sc.note("strings", "S", D - 12, t0, 2 * BAR, 110)
    sc.note("strings", "M2", D - 12, t0, 2 * BAR, 100)

    # ANSWER: his ten seconds
    t0 = starts["answer"]
    sc.phrase("veena", CALL[0], D, t0, 108)                                   # bar 1: the veena alone
    sc.hit("drum_kit", t0, KIT_KICK, ST * 2, 100)
    b2 = t0 + BAR                                                             # bar 2: a beat of dead air, then a 16th run
    for i, sw in enumerate(("S", "R2", "G2", "M2", "P", "D1", "N2", "S+", "R2+", "G2+", "M2+", "P+")):
        sc.note("veena", sw, D, b2 + (4 + i) * ST, ST * 0.9, 80 + 3 * i)
    b3 = t0 + 2 * BAR                                                         # bar 3: the stab, then the call two steps LATE
    for sw in ("S", "G2", "P"):
        sc.note("strings", sw, D, b3, ST * 3, 118)
    sc.hit("drum_kit", b3, KIT_KICK, ST * 2, 124)
    sc.hit("drum_kit", b3, KIT_CLAP, ST, 120)
    sc.phrase("veena", CALL[1], D, b3, 104, late=2)          # two steps late, so its last S+ falls off the bar: left hanging
    for s in range(2, STEPS, 4):
        sc.hit("kanjira", b3 + s * ST, 49, ST * 0.5, 70)
    b4 = t0 + 3 * BAR                                                         # bar 4: tresillo stop-time, then a roll
    for k, s in enumerate((0, 3, 6)):
        chord = [_step(raaga, sw, k) for sw in ("S", "G2", "P")]
        for sw in chord:
            sc.note("strings", sw, D, b4 + s * ST, ST * 2, 110 + 5 * k)
        sc.note("saxophone", chord[-1], D, b4 + s * ST, ST * 2, 110 + 5 * k)
        sc.hit("drum_kit", b4 + s * ST, KIT_KICK, ST * 2, 120)
        sc.hit("drum_kit", b4 + s * ST, KIT_CLAP, ST, 116)
    for s in range(8, STEPS):
        sc.hit("drum_kit", b4 + s * ST, KIT_CLAP, ST * 0.6, 50 + 9 * (s - 8))
    b5 = t0 + 4 * BAR                                                         # bar 5: the tihai, 3 x (P D1 N2), lands on sam
    sc.hit("mridangam", b5, 48, ST * 2, 110)
    sc.note("bass", "S", D - 24, b5, ST * 2, 112)
    for unit, start in enumerate((3, 8, 13)):
        for i, sw in enumerate(("P", "D1", "N2")):
            s = start + i
            sc.note("veena", sw, D + 12 * (unit == 2), b5 + s * ST, ST * 0.95, 96 + 8 * unit)
            sc.hit("mridangam", b5 + s * ST, 48 + i, ST, 92 + 10 * unit)
            sc.hit("drum_kit", b5 + s * ST, KIT_CLAP if i == 2 else KIT_HAT, ST * 0.6, 70 + 14 * unit)
    b6 = t0 + 5 * BAR                                                         # bar 6: SAM - everything - then air, then pick-up
    for sw in ("S", "P", "S+"):
        sc.note("strings", sw, D, b6, ST * 4, 127)
    sc.note("veena", "S+", D + 12, b6, ST * 4, 124)
    sc.note("saxophone", "S+", D, b6, ST * 4, 124)
    sc.note("bass", "S", D - 24, b6, ST * 4, 127)
    for m in (KIT_KICK, KIT_CLAP, KIT_HAT):
        sc.hit("drum_kit", b6, m, ST * 6, 127)
    sc.hit("mridangam", b6, 48, ST * 3, 124)
    for i, (s, sw) in enumerate(((8, "N2"), (9, "D1"), (10, "P"), (12, "D1"), (13, "N2"), (14, "S+"))):
        sc.note("veena", sw, D, b6 + s * ST, ST * 0.95, 100 + 3 * i)
        sc.note("saxophone", sw, D, b6 + s * ST, ST * 0.95, 96 + 3 * i)
        sc.hit("drum_kit", b6 + s * ST, KIT_CLAP, ST * 0.6, 84 + 6 * i)

    # GROOVE A and GROOVE B
    def groove(name: str, tonic: int, big: bool) -> None:
        g0 = starts[name]
        last_kick = None
        for bar in range(8):
            t0 = g0 + bar * BAR
            kicks = FOUR if big and bar % 2 == 0 else KICKS[int(sc.rng.integers(len(KICKS)))]
            if kicks == last_kick:                                       # never the same bar twice in a row
                kicks = KICKS[(KICKS.index(kicks) + 1) % len(KICKS)] if kicks in KICKS else KICKS[0]
            last_kick = kicks
            sc.kit_bar(t0, kicks, fill=bar % 4 == 3, open_hat=bar % 2 == 1, intensity=0.9 if big else 0.6)
            sc.bass_bar(t0, bar, tonic, 108 if big else 100)
            for s in (2, 6, 10, 14):
                sc.hit("tambourine", t0 + s * ST, 61, ST * 0.6, 62 if big else 52)
            second = bar >= 4
            k = bar % 4
            if k < 2:                                                    # the veena calls
                sc.phrase("veena", CALL[k], tonic, t0, 104, octave=1 if (big and second) else 0)
                if second:
                    sc.phrase("violin", CALL[k], tonic, t0, 66, up=2)    # a third above, along the raaga
            else:                                                        # the saxophone answers
                bar_notes = ANSWER_UP if (k == 3 and second) else ANSWER[k - 2]
                sc.phrase("saxophone", bar_notes, tonic, t0, 106)
                if big:
                    sc.phrase("violin", bar_notes, tonic, t0, 70, octave=1)
            sc.note("synth_pad", "S", tonic - 12, t0, BAR, 34)
            sc.note("synth_pad", "P", tonic - 12, t0, BAR, 28)
            if big:
                for s in (0, 3, 6, 8, 11, 14):
                    sc.hit("mridangam", t0 + s * ST, 48 + (s % 4), ST, 76 if s in (0, 8) else 58)
                sc.note("choir", "S", tonic, t0, BAR, 36 + 4 * bar)          # the crowd's "aa", rising
                sc.note("choir", "P", tonic, t0, BAR, 30 + 4 * bar)
                if bar >= 4:                                             # whistles
                    for _ in range(2):
                        s = int(sc.rng.integers(0, STEPS - 2))
                        sc.note("flute", "P+" if sc.rng.random() < 0.5 else "S+", tonic + 12, t0 + s * ST, ST * 1.5, 60)
        sc.tonic_spans.append((g0, g0 + 8 * BAR, tonic))

    sc.tonic_spans.append((0.0, starts["groove_a"], D))
    groove("groove_a", D, big=False)

    # BREAK: no kick; claps, mridangam then tabla, a veena korvai, a snare ramp
    t0 = starts["break"]
    for b in range(2):
        for s in (4, 12):
            sc.hit("drum_kit", t0 + b * BAR + s * ST, KIT_CLAP, ST, 104)
        drum = "mridangam" if b == 0 else "tabla"
        for s, stroke, v in ((0, 0, 100), (2, 2, 70), (3, 1, 80), (6, 3, 78), (8, 0, 96), (10, 2, 66), (11, 1, 74),
                             (13, 3, 80), (14, 2, 70), (15, 1, 86)):
            sc.hit(drum, t0 + b * BAR + s * ST, 48 + stroke, ST, v)
    sc.note("bass", "S", D - 24, t0, BAR * 1.5, 96)
    korvai = ("S+", "N2", "D1", "P", "D1", "N2", "S+", "R2+")
    for i in range(16):
        sc.note("veena", korvai[i % len(korvai)], D, t0 + i * ST * 1.5, ST * 1.3, 88 + (i % 4) * 6)
    for s in range(8, STEPS):
        sc.hit("drum_kit", t0 + BAR + s * ST, KIT_CLAP, ST * 0.6, 50 + 9 * (s - 8))
    sc.tonic_spans.append((starts["break"], starts["groove_b"], D))

    groove("groove_b", E, big=True)

    # TIHAI: the phrase three times with everyone, the final Sa, whistles over the ring-out
    t0 = starts["tihai"]
    unit = (("S+", 0), ("N2", 1), ("D1", 2), ("P", 3))
    for u, start in enumerate((0, 6, 12)):
        for sw, d in unit:
            s = start + d
            tt = t0 + s * ST
            for key, vel in (("veena", 104), ("saxophone", 104), ("violin", 80)):
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
    sc.tonic_spans.append((starts["tihai"], total + 2.0, E))
    return sc.parts, total, sc.tonic_spans


MIX = {"veena": (1.0, 0.15), "saxophone": (0.9, -0.08), "violin": (0.5, -0.25), "strings": (0.5, 0.0), "choir": (0.6, 0.0),
       "flute": (0.4, 0.3), "bass": (0.9, 0.0), "drum_kit": (1.0, 0.0), "mridangam": (0.6, -0.18), "tabla": (0.6, 0.2),
       "kanjira": (0.35, -0.3), "tambourine": (0.3, 0.32), "synth_pad": (0.18, 0.0)}


def render(sr: int = 44100, seed: int = 4, lead_in: float = 0.0) -> np.ndarray:
    """The piece as stereo audio; ``lead_in`` seconds of quiet first (Krish's get-ready rule when it is played straight)."""
    parts, seconds, _ = build(seed)
    total = lead_in + seconds + 2.5
    mix = dsp.silence(total, sr)
    for k, (key, notes) in enumerate(sorted(parts.items())):
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        shifted = [Note(swara=n.swara, midi=n.midi, start=round(n.start + lead_in, 4), duration=n.duration, velocity=n.velocity)
                   for n in notes]
        audio = render_notes(shifted, inst, sr, total_seconds=total, seed=seed + k)
        gain, pan = MIX.get(key, (0.6, 0.0))
        mix = dsp.mix_into(mix, dsp.pan_mono(audio * gain, pan), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.5, sr).astype(np.float32)
