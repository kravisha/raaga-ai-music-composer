"""Singing voice renderer (spec section 4 steps 5-6).

A source-filter singing synthesiser: a glottal source follows the melody's
pitch curve (with portamento, vibrato and gamaka), and a bank of formant
resonators shapes it into the vowel of each sung syllable.  Consonants are
short shaped transients at syllable onsets.

This is the local engine.  A cloud singing-synthesis or authorised
voice-conversion provider plugs in behind the same call in
:mod:`raagacomposer.providers` without the rest of the app changing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from ..core.models import (LyricsVersion, MelodyVersion, Note, VocalDirection,
                           VoiceProfile)
from ..music.theory import midi_to_freq

# Reference male formants (Hz). A profile's formant_shift scales these.
VOWEL_FORMANTS: Dict[str, Tuple[float, float, float, float]] = {
    "a": (730, 1090, 2440, 3400),
    "aa": (750, 1150, 2450, 3400),
    "e": (530, 1840, 2480, 3500),
    "ae": (660, 1720, 2410, 3400),
    "i": (270, 2290, 3010, 3600),
    "ee": (280, 2350, 3050, 3600),
    "o": (570, 840, 2410, 3300),
    "oo": (300, 870, 2240, 3200),
    "u": (320, 900, 2240, 3200),
    "ai": (600, 1700, 2500, 3400),
    "au": (620, 1000, 2400, 3300),
    # A closed hum, which is not a vowel at all: the lips are shut and the
    # sound leaves through the nose.  Its first resonance is low and the
    # rest are barely there, which is why a hum reads as warm and dark
    # where an open "aa" reads as bright.
    "hum": (280, 1100, 2000, 2800),
}

#: Per-formant loudness.  Sung vowels share the default; a hum does not,
#: and that is the whole difference between the two.  Open "aa" through
#: these gains sounds like a reed instrument, because a sawtooth with four
#: strong formants *is* roughly how you synthesise one.  Shutting the upper
#: formants down is what makes it sound like a closed mouth.
FORMANT_GAINS = (1.0, 0.62, 0.34, 0.18)
VOWEL_GAINS: Dict[str, Tuple[float, float, float, float]] = {
    "hum": (1.0, 0.22, 0.06, 0.02),
}
FORMANT_BW = (80.0, 110.0, 160.0, 220.0)

PLOSIVES = set("kgtdpb") | {"ch", "j", "tt", "dd"}
FRICATIVES = set("sfhvz") | {"sh", "th"}
NASALS = set("mn") | {"ng", "ny"}
LIQUIDS = set("lrywv")

STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "soft":       {"intensity": 0.35, "vibrato": 0.35, "breath": 0.8, "attack": 0.055},
    "intimate":   {"intensity": 0.35, "vibrato": 0.3, "breath": 0.85, "attack": 0.06},
    "strong":     {"intensity": 0.9, "vibrato": 0.5, "breath": 0.25, "attack": 0.018},
    "emotional":  {"intensity": 0.75, "vibrato": 0.8, "breath": 0.5, "attack": 0.035},
    "romantic":   {"intensity": 0.55, "vibrato": 0.6, "breath": 0.6, "attack": 0.04},
    "sad":        {"intensity": 0.45, "vibrato": 0.7, "breath": 0.7, "attack": 0.05},
    "energetic":  {"intensity": 0.95, "vibrato": 0.4, "breath": 0.2, "attack": 0.015},
    "devotional": {"intensity": 0.6, "vibrato": 0.55, "breath": 0.45, "attack": 0.045},
    "smooth":     {"intensity": 0.5, "vibrato": 0.45, "breath": 0.5, "attack": 0.05},
    "dramatic":   {"intensity": 0.85, "vibrato": 0.75, "breath": 0.35, "attack": 0.025},
}


@dataclass
class SungSegment:
    start: float
    end: float
    midi: int
    syllable: str = ""
    vowel: str = "a"
    consonant: str = ""
    velocity: int = 90
    gamaka: str = ""
    legato: bool = False
    coda: str = ""


def split_syllable(syllable: str) -> Tuple[str, str, str]:
    """Return (onset consonant, vowel, coda consonant) for a syllable.

    The coda used to be discarded.  "vaan" was sung "vaa" and "kal" was
    sung "ka", and a large part of what makes a word recognisable is at
    its end - so the words came out as vowels with a scratch in front of
    them.  This returns three parts now rather than two; the callers are
    few and the third one is the point.
    """
    s = re.sub(r"[^a-zA-Z]", "", (syllable or "")).lower()
    if not s:
        return "", "a", ""
    m = re.match(r"^([bcdfghjklmnpqrstvwxyz]{1,2})?(.*)$", s)
    cons = (m.group(1) or "") if m else ""
    rest = (m.group(2) or "") if m else s
    vowels = re.findall(r"(aa|ee|oo|ai|au|ae|[aeiou])", rest)
    vowel = vowels[0] if vowels else "a"
    coda = ""
    if vowels:
        after = rest[rest.index(vowel) + len(vowel):]
        # Only what closes this syllable: a following vowel means another
        # syllable was written into one slot, and guessing where to split
        # it would put sounds on notes nobody wrote them for.
        tail = re.match(r"^([bcdfghjklmnpqrstvwxyz]{1,2})(?![a-z])", after)
        coda = tail.group(1) if tail else ""
    if vowel not in VOWEL_FORMANTS:
        vowel = vowel[0] if vowel and vowel[0] in VOWEL_FORMANTS else "a"
    return cons, vowel, coda


def plan_segments(melody: MelodyVersion,
                  lyrics: Optional[LyricsVersion] = None,
                  vocal_sections_only: bool = True,
                  vowel: str = "") -> List[SungSegment]:
    """Map notes (and their fitted syllables) onto singable segments.

    ``vowel`` overrides what is sung, which is how a tune is hummed rather
    than sung open on "aa": every note takes the same closed sound and no
    consonant, because a hum has no words to shape.
    """
    syllable_for: Dict[int, str] = {}
    if lyrics:
        for line in lyrics.lines:
            for idx, syl in zip(line.note_indices, line.syllables):
                syllable_for[idx] = syl

    segments: List[SungSegment] = []
    prev_end = -1.0
    for i, note in enumerate(melody.notes):
        section = melody.section_by_id(note.section_id)
        if vocal_sections_only and section and section.kind.instrumental:
            continue
        syl = syllable_for.get(i, "")
        if vowel:
            cons, sound, coda = "", vowel, ""
        else:
            cons, sound, coda = (split_syllable(syl) if syl
                                 else ("", "a", ""))
        segments.append(SungSegment(
            start=note.start, end=note.end, midi=note.midi, syllable=syl,
            vowel=sound, consonant=cons, coda=coda, velocity=note.velocity,
            gamaka=note.gamaka, legato=(note.start - prev_end) < 0.06))
        prev_end = note.end
    return segments


def _resonator(freq: float, bw: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    freq = max(80.0, min(freq, sr * 0.45))
    r = float(np.exp(-np.pi * bw / sr))
    theta = 2 * np.pi * freq / sr
    a = np.array([1.0, -2 * r * np.cos(theta), r * r], dtype=np.float64)
    b = np.array([1.0 - r, 0.0, 0.0], dtype=np.float64)
    return b, a


def _style(direction: VocalDirection) -> Dict[str, float]:
    preset = dict(STYLE_PRESETS.get((direction.style or "romantic").lower(),
                                    STYLE_PRESETS["romantic"]))
    preset["intensity"] = float(np.clip(
        0.5 * preset["intensity"] + 0.5 * direction.intensity, 0.05, 1.0))
    preset["vibrato"] = float(np.clip(
        0.5 * preset["vibrato"] + 0.5 * direction.vibrato, 0.0, 1.0))
    preset["breath"] = float(np.clip(
        0.5 * preset["breath"] + 0.5 * direction.breath, 0.0, 1.0))
    return preset


def render(segments: Sequence[SungSegment], profile: VoiceProfile,
           direction: VocalDirection, sr: int = 44100,
           total_seconds: Optional[float] = None,
           seed: int = 11) -> np.ndarray:
    """Render a mono vocal line."""
    if not segments:
        return np.zeros(int((total_seconds or 0.0) * sr), dtype=np.float32)

    style = _style(direction)
    rng = np.random.default_rng(seed)
    end = total_seconds if total_seconds is not None else \
        max(s.end for s in segments) + 0.6
    n = max(1, int(end * sr))

    f0 = np.zeros(n, dtype=np.float32)
    amp = np.zeros(n, dtype=np.float32)
    voiced = np.zeros(n, dtype=bool)

    attack = style["attack"] * (1.4 - 0.6 * direction.phrase_emphasis)
    release = 0.05 + 0.12 * direction.sustain

    spans: List[Tuple[int, int, SungSegment]] = []
    for seg in segments:
        a = int(seg.start * sr)
        b = min(n, int(seg.end * sr))
        if b <= a:
            continue
        spans.append((a, b, seg))
        f0[a:b] = midi_to_freq(seg.midi)
        voiced[a:b] = True

        length = b - a
        env = np.ones(length, dtype=np.float32)
        na = min(length, max(8, int(attack * sr)))
        env[:na] = np.linspace(0.0, 1.0, na, dtype=np.float32) ** 1.4
        nr = min(length - na, max(8, int(release * sr)))
        if nr > 0:
            env[-nr:] *= np.linspace(1.0, 0.15, nr, dtype=np.float32)
        level = 0.35 + 0.65 * (seg.velocity / 110.0) * style["intensity"]
        level *= 0.85 + 0.3 * direction.dynamics * rng.random()
        amp[a:b] = np.maximum(amp[a:b], env * level)

    # Portamento between adjacent notes.
    for i in range(1, len(spans)):
        a0, b0, s0 = spans[i - 1]
        a1, b1, s1 = spans[i]
        gap = (s1.start - s0.end)
        if gap > 0.14 or a1 <= b0 - 1:
            continue
        glide = min(int(0.07 * sr), (b1 - a1) // 2)
        if glide > 2:
            f0[a1:a1 + glide] = np.linspace(midi_to_freq(s0.midi),
                                            midi_to_freq(s1.midi), glide)
            if gap > 0:
                f0[b0:a1] = midi_to_freq(s0.midi)
                voiced[b0:a1] = True
                amp[b0:a1] = amp[b0 - 1] if b0 > 0 else 0.0

    # Fill unvoiced gaps so the phase accumulator stays continuous.
    f0[f0 <= 0] = midi_to_freq(profile.base_midi)

    t = np.arange(n, dtype=np.float32) / sr
    # Vibrato: profile rate, depth scaled by direction, only on longer notes.
    vib_depth = profile.vibrato_depth * (0.4 + 1.2 * style["vibrato"])
    vib_gate = np.zeros(n, dtype=np.float32)
    for a, b, seg in spans:
        dur = (b - a) / sr
        if dur < 0.28:
            continue
        onset = a + int(min(0.18, dur * 0.35) * sr)
        if onset < b:
            ramp = np.linspace(0.0, 1.0, b - onset, dtype=np.float32)
            vib_gate[onset:b] = ramp
    vibrato = np.sin(2 * np.pi * profile.vibrato_rate * t).astype(np.float32)
    f0 = f0 * (2 ** (vib_depth * vibrato * vib_gate / 12.0))

    # Gamaka: extra oscillation on marked notes.
    for a, b, seg in spans:
        g = (seg.gamaka or "").lower()
        if not g or b - a < int(0.12 * sr):
            continue
        tt = np.arange(b - a, dtype=np.float32) / sr
        if g.startswith("kampita"):
            f0[a:b] *= 2 ** (0.45 * np.sin(2 * np.pi * 5.5 * tt) / 12.0)
        elif g.startswith("slide_up"):
            k = max(2, int((b - a) * 0.3))
            f0[a:a + k] *= np.linspace(2 ** (-1.6 / 12), 1.0, k)
        elif g.startswith("slide_down"):
            k = max(2, int((b - a) * 0.3))
            f0[a:a + k] *= np.linspace(2 ** (1.6 / 12), 1.0, k)

    # Jitter keeps it from sounding like an oscillator.
    jitter = rng.standard_normal(n // 512 + 2).astype(np.float32)
    jitter = np.interp(np.arange(n), np.linspace(0, n, len(jitter)), jitter)
    f0 = f0 * (1.0 + 0.0025 * jitter).astype(np.float32)

    # Glottal source: band-limited-ish sawtooth with spectral tilt.
    phase = np.cumsum(f0) / sr
    saw = (2.0 * (phase - np.floor(phase)) - 1.0).astype(np.float32)
    tilt = float(np.exp(-2 * np.pi * 900.0 / sr))
    source = lfilter([1 - tilt], [1, -tilt], saw).astype(np.float32)
    source = source * 2.2
    breath_level = profile.breathiness * (0.4 + 1.2 * style["breath"])
    source = source + rng.standard_normal(n).astype(np.float32) * breath_level * 0.35

    # Formant filtering, one segment at a time with state carried over.
    out = np.zeros(n, dtype=np.float32)
    shift = profile.formant_shift
    states = [None] * 4
    last_end = 0
    for a, b, seg in spans:
        if a > last_end:
            last_end = a
        seg_src = source[a:b]
        if len(seg_src) == 0:
            continue
        formants = VOWEL_FORMANTS.get(seg.vowel, VOWEL_FORMANTS["a"])
        gains = VOWEL_GAINS.get(seg.vowel, FORMANT_GAINS)
        mixed = np.zeros(len(seg_src), dtype=np.float32)
        for k, (f, gain, bw) in enumerate(zip(formants, gains, FORMANT_BW)):
            freq = f * shift * (1.0 + 0.06 * (profile.brightness - 1.0) * k)
            bnum, aden = _resonator(freq, bw * (1.0 + 0.3 * k), sr)
            zi = states[k]
            if zi is None:
                zi = lfilter_zi(bnum, aden) * float(seg_src[0])
            y, zf = lfilter(bnum, aden, seg_src, zi=zi)
            states[k] = zf
            mixed += (y * gain).astype(np.float32)
        peak = float(np.abs(mixed).max())
        if peak > 0:
            mixed /= peak
        out[a:b] += mixed * amp[a:b]

    # Consonant onsets.
    for a, b, seg in spans:
        if not seg.consonant:
            continue
        out = _add_consonant(out, seg.consonant, a, sr, rng,
                             level=0.35 * style["intensity"] + 0.1)

    # Codas.  A closing consonant is not something added on top of a
    # vowel - it is the vowel stopping.  Measured on one note, adding it
    # over the top left it about eighteen decibels under the vowel and
    # inaudible, which is a coda that exists in the buffer and not in the
    # room.  So the vowel is faded across the coda's span and the
    # consonant is set against how loud the voice actually is there,
    # rather than against a constant.
    for a, b, seg in spans:
        if not seg.coda:
            continue
        out = _close_with_consonant(out, seg.coda, a, b, sr, rng,
                                    style["intensity"])

    # Breaths in the gaps between phrases.
    if style["breath"] > 0.3:
        for i in range(1, len(spans)):
            gap_start = spans[i - 1][1]
            gap_end = spans[i][0]
            if gap_end - gap_start < int(0.18 * sr):
                continue
            length = min(int(0.22 * sr), gap_end - gap_start)
            at = max(gap_start, gap_end - length)
            noise = rng.standard_normal(length).astype(np.float32)
            env = np.hanning(length).astype(np.float32)
            out[at:at + length] += noise * env * 0.02 * style["breath"]

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * 0.85
    return out.astype(np.float32)


def _close_with_consonant(buf: np.ndarray, cons: str, a: int, b: int,
                          sr: int, rng: np.random.Generator,
                          intensity: float) -> np.ndarray:
    """End a note on its closing consonant, by closing the vowel into it."""
    length = _consonant_length(cons, sr)
    if length < 4:
        return buf
    start = max(a, b - length)
    length = min(length, b - start)
    if length < 4 or start + length > len(buf):
        return buf
    voice = float(np.sqrt(np.mean(buf[start:start + length] ** 2)))
    # The mouth closes: the vowel falls away rather than continuing under
    # the consonant.  Not to silence - a nasal is voiced, and cutting it
    # dead sounds like an edit.
    fade = np.linspace(1.0, 0.3, length).astype(np.float32)
    buf[start:start + length] *= fade
    level = max(0.05, voice * (0.9 + 0.5 * intensity))
    return _add_consonant(buf, cons, b, sr, rng, level=level,
                          trailing=True, limit=a)


def _consonant_length(cons: str, sr: int) -> int:
    c = cons[:2] if cons[:2] in PLOSIVES | FRICATIVES | NASALS else cons[:1]
    if c in PLOSIVES:
        return int(0.028 * sr)
    if c in FRICATIVES:
        return int(0.075 * sr)
    if c in NASALS:
        return int(0.06 * sr)
    if c in LIQUIDS:
        return int(0.045 * sr)
    return 0


def _add_consonant(buf: np.ndarray, cons: str, at: int, sr: int,
                   rng: np.random.Generator, level: float = 0.3,
                   trailing: bool = False, limit: int = 0) -> np.ndarray:
    """Put a consonant at *at*.

    ``trailing`` places it ending at *at* rather than beginning there,
    which is what a coda is: the close of the note it belongs to.
    ``limit`` is that note's start, so a short note cannot have its
    closing consonant pushed back over the note before it.
    """
    c = cons[:2] if cons[:2] in PLOSIVES | FRICATIVES | NASALS else cons[:1]
    if c in PLOSIVES:
        dur, colour, gap = 0.028, 4200.0, 0.012
    elif c in FRICATIVES:
        dur, colour, gap = 0.075, 6500.0, 0.0
    elif c in NASALS:
        dur, colour, gap = 0.06, 400.0, 0.0
    elif c in LIQUIDS:
        dur, colour, gap = 0.045, 1400.0, 0.0
    else:
        return buf
    length = int(dur * sr)
    if trailing:
        start = max(limit, at - length)
    else:
        start = max(0, at - int((dur + gap) * sr))
    if start + length >= len(buf) or length < 4 or start < 0:
        return buf
    noise = rng.standard_normal(length).astype(np.float32)
    alpha = float(np.exp(-2 * np.pi * colour / sr))
    low = lfilter([1 - alpha], [1, -alpha], noise).astype(np.float32)
    sig = low if c in NASALS or c in LIQUIDS else (noise - low)
    env = np.exp(-np.linspace(0, 4, length)).astype(np.float32)
    if c in NASALS or c in LIQUIDS:
        env = np.hanning(length).astype(np.float32)
    buf[start:start + length] += sig * env * level
    return buf


def render_melody(melody: MelodyVersion, lyrics: Optional[LyricsVersion],
                  profile: VoiceProfile, direction: VocalDirection,
                  sr: int = 44100, total_seconds: Optional[float] = None,
                  seed: int = 11,
                  vocal_sections_only: bool = True,
                  vowel: str = "",
                  section_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """Sing a melody, with words or without them.

    ``vocal_sections_only`` is right for a take with lyrics - nobody sings
    over the interlude - and wrong for hearing the tune itself, where every
    note has to sound.  On a 53-note tune it was the difference between the
    whole line and 33 notes with holes where the prelude, interlude and
    outro should be.

    ``section_ids`` sings only those sections, in their place in the song,
    so a creator settling the Pallavi hears the Pallavi rather than the
    whole take.  The rest of the timeline stays silent at full length: the
    take still lines up with the arrangement, which is what lets the two be
    heard together.
    """
    if section_ids:
        segments = _segments_for_sections(melody, lyrics, section_ids, vowel)
    else:
        segments = plan_segments(melody, lyrics,
                                 vocal_sections_only=vocal_sections_only,
                                 vowel=vowel)
    return render(segments, profile, direction, sr,
                  total_seconds or (melody.duration + 1.0), seed)


def _segments_for_sections(melody: MelodyVersion,
                           lyrics: Optional[LyricsVersion],
                           section_ids: Sequence[str],
                           vowel: str = "") -> List[SungSegment]:
    """The sung segments belonging to the chosen sections, and no others.

    Planned unfiltered so there is one segment per note and the two can be
    walked together - a segment does not carry its section, and the note
    does.  Instrumental sections stay silent unless they were chosen: this
    answers "sing the Pallavi", not "sing everything the singer could".
    """
    wanted = set(section_ids)
    sections = {section.id: section for section in melody.sections}
    planned = plan_segments(melody, lyrics, vocal_sections_only=False,
                            vowel=vowel)
    kept: List[SungSegment] = []
    previous_end = -1.0
    for note, segment in zip(melody.notes, planned):
        if note.section_id not in wanted:
            continue
        section = sections.get(note.section_id)
        if section is None:
            continue
        # A note excluded from the take is not a preceding sung note, so a
        # note that follows a gap must not slur into silence.
        segment.legato = (segment.start - previous_end) < 0.06
        kept.append(segment)
        previous_end = segment.end
    return kept
