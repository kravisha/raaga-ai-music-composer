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
}

FORMANT_GAINS = (1.0, 0.62, 0.34, 0.18)
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


def split_syllable(syllable: str) -> Tuple[str, str]:
    """Return (onset consonant, vowel) for a transliterated syllable."""
    s = re.sub(r"[^a-zA-Z]", "", (syllable or "")).lower()
    if not s:
        return "", "a"
    m = re.match(r"^([bcdfghjklmnpqrstvwxyz]{1,2})?(.*)$", s)
    cons = (m.group(1) or "") if m else ""
    rest = (m.group(2) or "") if m else s
    vowels = re.findall(r"(aa|ee|oo|ai|au|ae|[aeiou])", rest)
    vowel = vowels[0] if vowels else "a"
    if vowel not in VOWEL_FORMANTS:
        vowel = vowel[0] if vowel and vowel[0] in VOWEL_FORMANTS else "a"
    return cons, vowel


def plan_segments(melody: MelodyVersion,
                  lyrics: Optional[LyricsVersion] = None,
                  vocal_sections_only: bool = True) -> List[SungSegment]:
    """Map notes (and their fitted syllables) onto singable segments."""
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
        cons, vowel = split_syllable(syl) if syl else ("", "a")
        segments.append(SungSegment(
            start=note.start, end=note.end, midi=note.midi, syllable=syl,
            vowel=vowel, consonant=cons, velocity=note.velocity,
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
        mixed = np.zeros(len(seg_src), dtype=np.float32)
        for k, (f, gain, bw) in enumerate(zip(formants, FORMANT_GAINS, FORMANT_BW)):
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


def _add_consonant(buf: np.ndarray, cons: str, at: int, sr: int,
                   rng: np.random.Generator, level: float = 0.3) -> np.ndarray:
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
    start = max(0, at - int((dur + gap) * sr))
    length = int(dur * sr)
    if start + length >= len(buf) or length < 4:
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
                  seed: int = 11) -> np.ndarray:
    segments = plan_segments(melody, lyrics)
    return render(segments, profile, direction, sr,
                  total_seconds or (melody.duration + 1.0), seed)
