"""RAVANAN THIRUMBITTAAN - the song for Ravana's Answer II (Claude Work, 7 Nov 2026).

Krish, after hearing the composition: "It's beautiful. I like this composition. We need to have a nice song for this." - in Tamil.

The words sit on the veena hook one syllable to a note (the hook's bars have 7, 10, 7 and 7 notes; the flying answer has 9).
The story is the composition's: a young artist the crowd has booed asks for ten seconds, and turns the town around.

PALLAVI (on the hook)
  கேளு கேளு கேளுடா                 Kellu kellu kelluda                 Listen, listen, listen!
  காத்திருந்த நாட்கள் முடிந்தது       Kaathirundha naatkal mudindhadhu     The days of waiting are over
  ராவணன் திரும்பிட்டான்              Raavanan thirumbittaan              Ravana has come back
  கைதட்டும் நாள் வருது               Kaithattum naal varudhu             The day of applause is coming
  ஊரெல்லாம் எழுந்து ஆடுது            Oorellaam ezhundhu aadudhu          The whole town rises and dances  (the answer that flies up)
CHARANAM (the half-time drop, the hook at half speed)
  கூச்சல் போட்ட கூட்டமே              Koochal potta koottame              O crowd that jeered at me,
  பத்து நொடி கொடு என்னை பாரு          Pathu nodi kodu ennai paaru         give me ten seconds - watch me
TIHAI (the ending, three times)        Raa - va - nan - daa!

Sung by the composer's own synthesised voice (Kali's seat: female; a male voice an octave down).  Krish's own voice comes when his
samples are recorded.  The Tamil was written by Claude Work; Krish, the Tamil speaker, has the last word on every line.
"""
from __future__ import annotations

from dataclasses import dataclass

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
from ..raaga.library import library
from ..voice import mastering
from . import answer2 as A2
from .answer import ANSWER, ANSWER_UP, BAR, BPM, CALL, RAAGA, ST, TONIC, _Score


@dataclass(frozen=True)
class Line:
    tamil: str
    roman: str          # syllables separated by "-" and spaces: one per note
    english: str

    @property
    def syllables(self) -> list[str]:
        return [s for word in self.roman.lower().split() for s in word.split("-") if s]


L1 = Line("கேளு கேளு கேளுடா", "kel-lu kel-lu kel-lu-daa", "Listen, listen, listen!")
L2 = Line("காத்திருந்த நாட்கள் முடிந்தது", "kaa-thi-run-dha naat-kal mu-din-dha-dhu", "The days of waiting are over")
L3 = Line("ராவணன் திரும்பிட்டான்", "raa-va-nan thi-rum-bit-taan", "Ravana has come back")
L4 = Line("கைதட்டும் நாள் வருது", "kai-that-tum naal va-ru-dhu", "The day of applause is coming")
L4_UP = Line("ஊரெல்லாம் எழுந்து ஆடுது", "oo-rel-laam e-zhun-dhu aa-du-dhu", "The whole town rises and dances")
C1 = Line("கூச்சல் போட்ட கூட்டமே", "koo-chal pot-ta koot-ta-mae", "O crowd that jeered at me,")
C2 = Line("பத்து நொடி கொடு என்னை பாரு", "pat-thu no-di ko-du en-nai paa-ru", "give me ten seconds - watch me")
TIHAI = Line("ராவணன்டா", "raa-va-nan-daa", "Ravana!")

#: which words go on which bar of the hook
WORDS = {id(CALL[0]): L1, id(CALL[1]): L2, id(ANSWER[0]): L3, id(ANSWER[1]): L4, id(ANSWER_UP): L4_UP}

PALLAVI = (L1, L2, L3, L4)
CHARANAM = (C1, C2)


def _sing(sc: _Score, sung: list[tuple[Note, str]], bar: tuple, line: Line, tonic: int, t0: float, vel: int,
          stretch: float = 1.0) -> None:
    syl = line.syllables
    if len(syl) != len(bar):
        raise ValueError(f"'{line.roman}' has {len(syl)} syllables for {len(bar)} notes")
    for (s, length, sw), word in zip(bar, syl):
        before = len(sc.parts.get("voice", []))
        sc.note("voice", sw, tonic, t0 + s * ST * stretch, length * ST * 0.95 * stretch, vel)
        sung.append((sc.parts["voice"][before], word))


def vocal_line(seed: int = 5) -> tuple[list[tuple[Note, str]], dict[str, float], float]:
    """Every sung note with its syllable, on the composition's own timeline. -> (notes, section starts, seconds)."""
    _parts, seconds, _spans, starts = A2.build(seed)
    sc = _Score(library().require(RAAGA), seed)
    sung: list[tuple[Note, str]] = []
    D, E = TONIC, TONIC + A2.LIFT
    t0 = starts["answer"]                                      # his ten seconds open with the pallavi
    for b, bar in enumerate((CALL[0], CALL[1], ANSWER[0], ANSWER[1])):
        _sing(sc, sung, bar, PALLAVI[b], D, t0 + b * BAR, 104)
    g0 = starts["groove_a"]                                    # groove A, straight half: the pallavi again
    for b, bar in enumerate((CALL[0], CALL[1], ANSWER[0], ANSWER[1])):
        _sing(sc, sung, bar, PALLAVI[b], D, g0 + b * BAR, 100)
    h0 = starts["half_time"]                                   # the charanam, at half speed
    for half in range(2):
        _sing(sc, sung, CALL[half], CHARANAM[half], D, h0 + half * 2 * BAR, 108, stretch=2.0)
    b0 = starts["groove_b"]                                    # groove B, up a whole step: call and answer, a bar each
    for bar in range(8):
        tb = b0 + bar * BAR
        if bar % 2 == 0:
            phrase = CALL[(bar // 2) % 2]
        else:
            phrase = ANSWER_UP if bar == 7 else ANSWER[(bar // 2) % 2]
        _sing(sc, sung, phrase, WORDS[id(phrase)], E, tb, 110)
    t0 = starts["tihai"]                                       # Raa-va-nan-daa, three times, then the last Sa held
    for start in (0, 6, 12):
        _sing(sc, sung, tuple((start + d, 1, sw) for d, sw in enumerate(("S+", "N2", "D1", "P"))), TIHAI, E, t0, 116)
    before = len(sc.parts["voice"])
    sc.note("voice", "S+", E, t0 + BAR, BAR * 0.9, 118)
    sung.append((sc.parts["voice"][before], "aa"))
    return sung, starts, seconds


def lyric_sheet() -> str:
    out = ["RAVANAN THIRUMBITTAAN (Ravana's Answer II)", "", "Pallavi"]
    out += [f"  {x.tamil}  |  {x.roman.replace('-', '')}  |  {x.english}" for x in (*PALLAVI, L4_UP)]
    out += ["", "Charanam"] + [f"  {x.tamil}  |  {x.roman.replace('-', '')}  |  {x.english}" for x in CHARANAM]
    out += ["", "Tihai", f"  {TIHAI.tamil} x3  |  Raa-va-nan-daa!  |  {TIHAI.english}"]
    return "\n".join(out)


def render(sr: int = 44100, seed: int = 5, lead_in: float = 0.0, voice: str = "female") -> np.ndarray:
    """The song: Ravana's Answer II with the words sung over it."""
    from ..providers.local import LocalVoiceProvider
    from ..voice.profiles import BUILTIN

    band = A2.render(sr=sr, seed=seed, lead_in=lead_in)
    sung, _starts, _seconds = vocal_line(seed)
    sung.sort(key=lambda p: p[0].start)
    profile = next((p for p in BUILTIN if p.gender == voice), BUILTIN[0])
    semis = -12 if profile.gender == "male" else 0
    notes = [Note(swara=n.swara, midi=n.midi + semis, start=round(n.start + lead_in, 4), duration=n.duration,
                  velocity=n.velocity) for n, _w in sung]
    total = len(band) / sr
    melody = MelodyVersion(raaga=RAAGA, tonic_midi=TONIC + semis, tempo_bpm=BPM, notes=notes,
                           sections=[Section(id="song", kind=SectionKind.PALLAVI, start=0.0, end=total)])
    lyrics = LyricsVersion(language="Tamil", lines=[LyricLine(section_id="song", text="song",
                                                              syllables=[w for _n, w in sung],
                                                              note_indices=list(range(len(sung))))])
    direction = VocalDirection(style="energetic")
    vocal = LocalVoiceProvider().render_vocal(melody, lyrics, profile, direction, sr, total, seed=seed + 100,
                                              vocal_sections_only=False)
    vocal = mastering.quick_preview(vocal, sr, direction)
    mix = dsp.mix_into(dsp.as_stereo(band) * 0.8, dsp.pan_mono(vocal, 0.0) * 1.0 if vocal.ndim == 1 else vocal, 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.5, sr).astype(np.float32)
