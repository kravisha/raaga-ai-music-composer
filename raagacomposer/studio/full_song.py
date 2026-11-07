"""RAVANAN THIRUMBITTAAN - the full song (Claude Work, 7 Nov 2026).

Krish: "Can you make it clean ... together as a solid song. We need a good start, words, end. It's nice. We got a good piece.
Keep it going."  His picks so far: the words (approved), veena and flute, bongos, and the ending ("powerful") kept as it is.

    INTRO        7 bars  one bar of boos; the hook on veena (call) and flute (answer) over stop-time; the tihai onto sam
    PALLAVI      8 bars  the four lines sung over the groove, twice; the second time the answer flies up
    INTERLUDE    4 bars  veena and flute alone with the groove: the call a beat late over 3-against-4 hats
    CHARANAM     6 bars  two lines at half speed over a half-time beat, then two new softer lines (mellinam) on the answer
    BREAK        2 bars  khanda nadai, fives on bongos and tabla, the veena in fives
    PALLAVI (up) 8 bars  a whole step up (D -> E), double-time hats, call and answer a bar each, whistles
    ENDING       2 bars  the tihai - "Raa-va-nan-daa!" three times - and the last Sa, as Krish liked it

The two new charanam lines lean soft, as Krish taught: mellinam for melody, vallinam for the fierce parts.
    வீணை பேசும் நேரம்தான்     Veenai pesum neramdhaan     Now it is the veena's turn to speak
    மௌனம் ஆனது ஊரு         Mounam aanadhu ooru         and the whole town fell silent
"""
from __future__ import annotations

import numpy as np

from ..audio import dsp
from ..core.models import (
    LyricLine,
    LyricsVersion,
    MelodyVersion,
    Note,
    Section,
    SectionKind,
    VocalDirection,
)
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import library
from ..voice import mastering
from .answer import (
    ANSWER,
    ANSWER_UP,
    BAR,
    BPM,
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
from .answer2 import _phrase
from .groove import _step
from .song import C1, C2, L1, L2, L3, L4, L4_UP, TIHAI, Line

C3 = Line("வீணை பேசும் நேரம்தான்", "vee-nai pae-sum nae-ram-dhaan", "Now it is the veena's turn to speak")
C4 = Line("மௌனம் ஆனது ஊரு", "mow-nam aa-na-dhu oo-ru", "and the whole town fell silent")
PALLAVI = (L1, L2, L3, L4)
WORDS = {id(CALL[0]): L1, id(CALL[1]): L2, id(ANSWER[0]): L3, id(ANSWER[1]): L4, id(ANSWER_UP): L4_UP}

SECTIONS = (("intro", 7), ("pallavi", 8), ("interlude", 4), ("charanam", 6), ("break", 2), ("pallavi_up", 8), ("ending", 2))
HAND = "bongos"                      # Krish's pick
ANSWER_INST = "flute"                # veena and flute
GAINS = {"veena": 1.3, "drum_kit": 0.6, HAND: 0.75}
FULL_MIX = {**MIX, ANSWER_INST: (0.86, -0.12), "whistle": (0.4, 0.3), HAND: (0.6, 0.18), "tabla": (0.55, -0.2)}


class _Song(_Score):
    def __init__(self, seed: int) -> None:
        super().__init__(library().require(RAAGA), seed)
        self.sung: list[tuple[Note, str]] = []

    def sing(self, bar: tuple, line: Line, tonic: int, t0: float, vel: int, stretch: float = 1.0) -> None:
        syl = line.syllables
        if len(syl) != len(bar):
            raise ValueError(f"'{line.roman}' has {len(syl)} syllables for {len(bar)} notes")
        for (s, length, sw), word in zip(bar, syl):
            self.note("voice", sw, tonic, t0 + s * ST * stretch, length * ST * 0.95 * stretch, vel)
            self.sung.append((self.parts["voice"][-1], word))

    def groove_bar(self, tb: float, bar: int, tonic: int, *, kicks, hats, hand_strokes=True, fill=False, vel=100) -> None:
        for s in kicks:
            self.hit("drum_kit", tb + s * ST, KIT_KICK, ST * 2, 116 if s == 0 else 96)
        for s in (4, 12):
            self.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 110)
        for s in hats:
            self.hit("drum_kit", tb + s * ST, KIT_HAT, ST * 0.4, 76 if s % 4 == 0 else 48)
        if hand_strokes:                                       # the bongos: macho on the off-beats, hembra on 12 and 14, a slap
            for s, stroke, v in ((0, 1, 92), (2, 1, 54), (4, 2, 82), (6, 1, 54), (8, 1, 88), (10, 1, 54), (12, 0, 82),
                                 (14, 0, 72), (15, 1, 50)):
                self.hit(HAND, tb + s * ST, 48 + stroke, ST, v)
        if fill:
            for i, s in enumerate((12, 13, 14, 15)):
                self.hit(HAND, tb + s * ST, 48 + min(i, 2), ST, 80 + 10 * i)
        self.bass_bar(tb, bar, tonic, vel)
        for s in (2, 6, 10, 14):
            self.hit("tambourine", tb + s * ST, 61, ST * 0.6, 50)


def _kicks(sc: _Song, last):
    k = KICKS[int(sc.rng.integers(len(KICKS)))]
    return KICKS[(KICKS.index(k) + 1) % len(KICKS)] if k == last else k


def build(seed: int = 7) -> tuple[dict[str, list[Note]], list[tuple[Note, str]], float, list[tuple[float, float, int]],
                                   dict[str, float]]:
    """-> (band parts, sung notes with syllables, seconds, [(start, end, tonic)], {section: start})."""
    sc = _Song(seed)
    raaga = sc.raaga
    starts: dict[str, float] = {}
    t = 0.0
    for name, bars in SECTIONS:
        starts[name] = t
        t += bars * BAR
    total = t
    D, E = TONIC, TONIC + LIFT

    # INTRO: the boos, then the hook on veena and flute over stop-time, the tihai, sam
    t0 = starts["intro"]
    for i, sw in enumerate(("P", "M2", "G2", "R2")):
        sc.note("choir", sw, D - 12, t0 + i * BAR / 4, BAR * 0.3, 124 - 6 * i)
        sc.note("choir", _step(raaga, sw, 2), D - 12, t0 + i * BAR / 4, BAR * 0.3, 112 - 6 * i)
    for s in (0, 3, 8, 11):
        sc.hit("drum_kit", t0 + s * ST, KIT_KICK, ST * 2, 118 if s in (0, 8) else 92)
    sc.note("strings", "S", D - 12, t0, BAR, 110)
    sc.note("strings", "M2", D - 12, t0, BAR, 100)
    h0 = t0 + BAR
    for b in range(2):
        _phrase(sc, "veena", CALL[b], D, h0 + b * BAR, 112)
        for s in (4, 12):
            sc.hit("drum_kit", h0 + b * BAR + s * ST, KIT_CLAP, ST, 96)
    for b in range(2):
        tb = h0 + (2 + b) * BAR
        _phrase(sc, ANSWER_INST, ANSWER[b], D, tb, 110)
        for k, s in enumerate((0, 3, 6) if b == 0 else (0, 3, 6, 10, 12, 14)):
            for sw in (_step(raaga, x, k % 3) for x in ("S", "G2", "P")):
                sc.note("strings", sw, D, tb + s * ST, ST * 2, 104 + 4 * k)
            sc.hit("drum_kit", tb + s * ST, KIT_KICK, ST * 2, 118)
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 112)
    b5 = h0 + 4 * BAR
    for unit, start in enumerate((3, 8, 13)):
        for i, sw in enumerate(("P", "D1", "N2")):
            s = start + i
            sc.note("veena", sw, D + 12 * (unit == 2), b5 + s * ST, ST * 0.95, 96 + 8 * unit)
            sc.hit(HAND, b5 + s * ST, 48 + i, ST, 92 + 10 * unit)
            sc.hit("drum_kit", b5 + s * ST, KIT_CLAP if i == 2 else KIT_HAT, ST * 0.6, 70 + 14 * unit)
    b6 = h0 + 5 * BAR
    for sw in ("S", "P", "S+"):
        sc.note("strings", sw, D, b6, ST * 4, 127)
    sc.note("veena", "S+", D + 12, b6, ST * 4, 124)
    sc.note("bass", "S", D - 24, b6, ST * 4, 127)
    for m in (KIT_KICK, KIT_CLAP, KIT_HAT):
        sc.hit("drum_kit", b6, m, ST * 6, 127)
    for i, (s, sw) in enumerate(((8, "N2"), (9, "D1"), (10, "P"), (12, "D1"), (13, "N2"), (14, "S+"))):
        sc.note("veena", sw, D, b6 + s * ST, ST * 0.95, 100 + 3 * i)
        sc.hit("drum_kit", b6 + s * ST, KIT_CLAP, ST * 0.6, 84 + 6 * i)

    # PALLAVI: sung twice over the groove; the veena doubles the call, the flute the answer
    p0 = starts["pallavi"]
    last = None
    for bar in range(8):
        tb = p0 + bar * BAR
        last = _kicks(sc, last)
        sc.groove_bar(tb, bar, D, kicks=last, hats=range(0, STEPS, 2), fill=bar % 4 == 3)
        k = bar % 4
        phrase = CALL[k] if k < 2 else (ANSWER_UP if bar == 7 else ANSWER[k - 2])
        _phrase(sc, "veena" if k < 2 else ANSWER_INST, phrase, D, tb, 96)
        sc.sing(phrase, WORDS[id(phrase)], D, tb, 106)
        sc.note("synth_pad", "S", D - 12, tb, BAR, 34)

    # INTERLUDE: veena and flute alone over the groove; the call a beat late over 3:4 hats
    i0 = starts["interlude"]
    for bar in range(4):
        tb = i0 + bar * BAR
        last = _kicks(sc, last)
        sc.groove_bar(tb, bar, D, kicks=last, hats=range(0, STEPS, 3), fill=bar == 3)
        if bar == 0:
            _phrase(sc, "veena", CALL[0], D, tb, 108, late=4)         # its tail lands in bar 2, under the veena's next call
        elif bar == 1:                                            # no spill here: it would clash with the flute's answer (review)
            _phrase(sc, "veena", tuple(n for n in CALL[1] if n[0] + 4 < STEPS), D, tb, 108, late=4)
        else:
            _phrase(sc, ANSWER_INST, ANSWER_UP if bar == 3 else ANSWER[0], D, tb, 106)
            _phrase(sc, "violin", ANSWER_UP if bar == 3 else ANSWER[0], D, tb, 62, octave=1)

    # CHARANAM: C1, C2 at half speed over a half-time beat; then C3, C4 on the answer, building back
    c0 = starts["charanam"]
    for bar in range(4):
        tb = c0 + bar * BAR
        sc.hit("drum_kit", tb, KIT_KICK, ST * 4, 120)
        sc.hit("drum_kit", tb + 8 * ST, KIT_CLAP, ST * 2, 114)
        for s in range(0, STEPS, 4):
            sc.hit("drum_kit", tb + s * ST, KIT_HAT, ST * 1.5, 56)
        for s in (0, 6, 8, 14):
            sc.hit(HAND, tb + s * ST, 48 + (0 if s in (0, 8) else 1), ST, 70)
        root = ("S", "S", "D1", "P")[bar]
        sc.note("bass", root, D - 24, tb, BAR * 0.95, 112)
        sc.note("strings", root, D - 12, tb, BAR, 66)
    for half, line in enumerate((C1, C2)):
        _phrase(sc, "veena", CALL[half], D, c0 + half * 2 * BAR, 80, stretch=2.0)
        _phrase(sc, "violin", CALL[half], D, c0 + half * 2 * BAR, 54, stretch=2.0, octave=1)
        sc.sing(CALL[half], line, D, c0 + half * 2 * BAR, 110, stretch=2.0)
    for b, (phrase, line) in enumerate(((ANSWER[0], C3), (ANSWER[1], C4))):
        tb = c0 + (4 + b) * BAR
        sc.groove_bar(tb, b + 2, D, kicks=(0, 6, 10), hats=range(0, STEPS, 2), hand_strokes=b == 1, vel=92)   # D1, P roots
        _phrase(sc, ANSWER_INST, phrase, D, tb, 90)
        sc.sing(phrase, line, D, tb, 104)

    # BREAK: fives - bongos then tabla, the veena in fives, a snare ramp
    k0 = starts["break"]
    five = ("S+", "N2", "D1", "P", "M2")
    for bar in range(2):
        tb = k0 + bar * BAR
        for s in (4, 12):
            sc.hit("drum_kit", tb + s * ST, KIT_CLAP, ST, 100)
        for g, start in enumerate((0, 5, 10)):
            length = 6 if g == 2 else 5
            for i in range(length):
                s = start + i
                drum = HAND if bar == 0 else "tabla"
                strokes = 3 if drum == HAND else 4
                sc.hit(drum, tb + s * ST, 48 + (i % strokes), ST, 110 if i == 0 else 64)
                if bar == 1 or g < 2:
                    sc.note("veena", five[i] if i < 5 else "S+", D, tb + s * ST, ST * 0.9, 104 if i == 0 else 82)
    sc.note("bass", "S", D - 24, k0, BAR * 2, 96)
    for s in range(8, STEPS):
        if s != 12:
            sc.hit("drum_kit", k0 + BAR + s * ST, KIT_CLAP, ST * 0.6, 50 + 9 * (s - 8))
    sc.tonic_spans.append((0.0, starts["pallavi_up"], D))

    # PALLAVI (up a whole step): call and answer a bar each, double-time hats, the crowd's "aa", whistles
    u0 = starts["pallavi_up"]
    for bar in range(8):
        tb = u0 + bar * BAR
        kicks = FOUR if bar % 2 == 0 else KICKS[int(sc.rng.integers(len(KICKS)))]
        sc.groove_bar(tb, bar, E, kicks=kicks, hats=range(STEPS), fill=bar % 4 == 3, vel=108)
        phrase = CALL[(bar // 2) % 2] if bar % 2 == 0 else (ANSWER_UP if bar == 7 else ANSWER[(bar // 2) % 2])
        if bar % 2 == 0:
            _phrase(sc, "veena", phrase, E, tb, 104, octave=1 if bar >= 4 else 0)
        else:
            _phrase(sc, ANSWER_INST, phrase, E, tb, 100)
            _phrase(sc, "violin", phrase, E, tb, 64, octave=1)
        sc.sing(phrase, WORDS[id(phrase)], E, tb, 112)
        sc.note("choir", "S", E, tb, BAR, 40 + 5 * bar)
        sc.note("choir", "P", E, tb, BAR, 34 + 5 * bar)
        if bar >= 4:
            for _ in range(2):
                s = int(sc.rng.integers(0, STEPS - 2))
                sc.note("whistle", "P+" if sc.rng.random() < 0.5 else "S+", E + 12, tb + s * ST, ST * 1.5, 60)

    # ENDING: the tihai, three times with everyone, the voice chanting Raa-va-nan-daa; the last Sa; whistles
    e0 = starts["ending"]
    for u, start in enumerate((0, 6, 12)):
        unit = tuple((start + d, 1, sw) for d, sw in enumerate(("S+", "N2", "D1", "P")))
        for d, (_s, _l, sw) in enumerate(unit):
            tt = e0 + (start + d) * ST
            for key, vel in (("veena", 106), (ANSWER_INST, 100), ("violin", 80)):
                sc.note(key, sw, E + (12 if key == "violin" else 0), tt, ST * 0.95, vel + 6 * u)
            sc.hit("drum_kit", tt, KIT_KICK if d == 0 else KIT_CLAP, ST, 104 + 6 * u)
            sc.hit(HAND, tt, 48 + min(d, 2), ST, 96 + 8 * u)
        sc.sing(unit, TIHAI, E, e0, 116)
    end = e0 + BAR
    for sw in ("S", "P", "S+"):
        sc.note("strings", sw, E, end, BAR * 0.9, 127)
    sc.note("veena", "S+", E, end, BAR * 0.9, 127)
    sc.note(ANSWER_INST, "S+", E, end, BAR * 0.9, 110)
    sc.note("bass", "S", E - 24, end, BAR * 0.9, 127)
    sc.note("choir", "S+", E, end, BAR, 80)
    sc.note("voice", "S+", E, end, BAR * 0.9, 118)
    sc.sung.append((sc.parts["voice"][-1], "aa"))
    for m in (KIT_KICK, KIT_CLAP, KIT_HAT):
        sc.hit("drum_kit", end, m, ST * 8, 127)
    for i in range(6):
        sc.note("whistle", ("S+", "P+", "S+", "P+", "S++", "S+")[i], E + 12, end + (2 + i * 2) * ST, ST * 1.6, 70 - 4 * i)
    sc.tonic_spans.append((starts["pallavi_up"], total + 2.0, E))
    band = {k: v for k, v in sc.parts.items() if k != "voice"}
    return band, sc.sung, total, sc.tonic_spans, starts


def lyric_sheet() -> str:
    def row(x: Line) -> str:
        return f"  {x.tamil}  |  {x.roman.replace('-', '')}  |  {x.english}"
    out = ["RAVANAN THIRUMBITTAAN - the full song", "", "Intro: the hook on veena and flute", "", "Pallavi (twice)"]
    out += [row(x) for x in (*PALLAVI, L4_UP)]
    out += ["", "Interlude: veena and flute", "", "Charanam"] + [row(x) for x in (C1, C2, C3, C4)]
    out += ["", "Break: in fives", "", "Pallavi (a whole step up)", "", f"Ending: {TIHAI.tamil} x3 - Raa-va-nan-daa! - and the last Sa"]
    return "\n".join(out)


VOICE_OVER_BAND_DB = 1.0        # where the voice sits against the band while it sings (review: it was left to chance)


def vocal_balance(band: np.ndarray, vocal: np.ndarray) -> float:
    """The gain that puts the voice VOICE_OVER_BAND_DB above the band, measured only where the voice is singing."""
    n = min(len(band), len(vocal))
    v = np.abs(vocal[:n]).max(axis=1) if vocal.ndim == 2 else np.abs(vocal[:n])
    singing = v > 1e-3
    if not singing.any():
        return 1.0
    rms_v = float(np.sqrt(np.mean(vocal[:n][singing] ** 2)))
    rms_b = float(np.sqrt(np.mean((band[:n][singing] * 0.8) ** 2)))
    return (rms_b / max(rms_v, 1e-9)) * 10 ** (VOICE_OVER_BAND_DB / 20)


def render(sr: int = 44100, seed: int = 7, lead_in: float = 0.0, voice: str = "female") -> np.ndarray:
    from ..providers.local import LocalVoiceProvider
    from ..voice.profiles import BUILTIN

    band, sung, seconds, _spans, _starts = build(seed)
    total = lead_in + seconds + 2.5
    mix = dsp.silence(total, sr)
    for k, (key, notes) in enumerate(sorted(band.items())):
        inst = catalog.get("flute" if key == "whistle" else key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        shifted = [Note(swara=n.swara, midi=n.midi, start=round(n.start + lead_in, 4), duration=n.duration, velocity=n.velocity)
                   for n in notes]
        gain, pan = FULL_MIX.get(key, (0.6, 0.0))
        gain *= GAINS.get(key, 1.0)
        mix = dsp.mix_into(mix, dsp.pan_mono(render_notes(shifted, inst, sr, total_seconds=total, seed=seed + k) * gain, pan), 0)
    band_audio = dsp.limiter(dsp.normalize_loudness(mix, sr, -15.0), -1.0, sr)
    profile = next((p for p in BUILTIN if p.gender == voice), BUILTIN[0])
    semis = -12 if profile.gender == "male" else 0
    notes = [Note(swara=n.swara, midi=n.midi + semis, start=round(n.start + lead_in, 4), duration=n.duration, velocity=n.velocity)
             for n, _w in sung]
    melody = MelodyVersion(raaga=RAAGA, tonic_midi=TONIC + semis, tempo_bpm=BPM, notes=notes,
                           sections=[Section(id="song", kind=SectionKind.PALLAVI, start=0.0, end=total)])
    lyrics = LyricsVersion(language="Tamil", lines=[LyricLine(section_id="song", text="song", syllables=[w for _n, w in sung],
                                                              note_indices=list(range(len(sung))))])
    direction = VocalDirection(style="energetic")
    vocal = LocalVoiceProvider().render_vocal(melody, lyrics, profile, direction, sr, total, seed=seed + 100,
                                              vocal_sections_only=False)
    vocal = mastering.quick_preview(vocal, sr, direction)
    vocal = vocal if vocal.ndim == 2 else dsp.as_stereo(vocal)
    vocal = vocal * vocal_balance(band_audio, vocal)
    out = dsp.mix_into(band_audio * 0.8, vocal, 0)
    out = dsp.limiter(dsp.normalize_loudness(out, sr, -14.0), -1.0, sr)
    return dsp.fade(out, 0.0, 1.5, sr).astype(np.float32)
