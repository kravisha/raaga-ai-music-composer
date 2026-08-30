"""Raaga-aware melody generator (spec sections 4 step 3 and 10).

The generator walks the raaga's own ladder rather than a chromatic scale, so
ascending motion only uses arohanam notes and descending motion only uses
avarohanam notes.  Characteristic phrases (prayogas) are injected verbatim,
phrases cadence on resting notes (nyasa), and gamaka marks come from the raaga
data.  Everything is seeded, so regenerating one section leaves the rest of the
tune bit-identical.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.models import (ApprovalState, MelodyVersion, Note, Section,
                           SectionKind)
from ..raaga.library import Raaga, parse_swara
from .structure import plan_sections, section_role
from .theory import beat_seconds, cycle_seconds

# Rhythm patterns expressed in beats; scaled to the cycle length.
RHYTHM_BANK: Dict[str, List[List[float]]] = {
    "slow": [[2, 2, 4], [4, 2, 2], [2, 1, 1, 4], [3, 1, 4], [2, 2, 2, 2]],
    "medium": [[2, 1, 1, 2, 2], [1, 1, 2, 2, 2], [2, 2, 1, 1, 2], [1, 1, 1, 1, 4],
               [2, 1, 1, 1, 1, 2]],
    "fast": [[1, 1, 1, 1, 2, 2], [1, 0.5, 0.5, 1, 1, 2, 2], [1, 1, 2, 1, 1, 2],
             [0.5, 0.5, 1, 1, 1, 1, 1, 2]],
}

CONTOURS = ("arch", "rise", "fall", "wave", "flat")


@dataclass
class MelodyOptions:
    tempo_bpm: int = 72
    beats_per_cycle: int = 8
    tonic_midi: int = 60
    voice_low: int = 52
    voice_high: int = 79
    intensity: float = 0.6
    seed: int = 1
    song_type: str = "film song"
    duration_target: float = 150.0
    ornament: float = 0.5           # 0..1 how much gamaka and prayoga to use


# --------------------------------------------------------------------------
# token helpers
# --------------------------------------------------------------------------
def token_midi(raaga: Raaga, token: str, tonic: int) -> int:
    return raaga.midi(token, tonic)


def clamp_token(raaga: Raaga, token: str, tonic: int, low: int, high: int) -> str:
    """Shift a token by whole octaves until its pitch fits the given range."""
    base, octave = parse_swara(token)
    for _ in range(6):
        midi = raaga.midi(base + _oct_marks(octave), tonic)
        if midi < low:
            octave += 1
        elif midi > high:
            octave -= 1
        else:
            break
    return base + _oct_marks(octave)


def _oct_marks(octave: int) -> str:
    if octave > 0:
        return "+" * octave
    if octave < 0:
        return "-" * -octave
    return ""


def _with_octave(base: str, octave: int) -> str:
    return base + _oct_marks(octave)


def _nearest_nyasa(raaga: Raaga, token: str) -> str:
    if not raaga.nyasa:
        return token
    base, octave = parse_swara(token)
    deg = raaga.degree(token)
    best, best_d = token, 99
    for n in raaga.nyasa:
        for o in (octave - 1, octave, octave + 1):
            cand = _with_octave(n, o)
            d = abs(raaga.degree(cand) - deg)
            if d < best_d:
                best_d, best = d, cand
    return best


# --------------------------------------------------------------------------
# phrase generation
# --------------------------------------------------------------------------
def _shape(kind: str, frac: float) -> float:
    import math
    if kind == "rise":
        return frac
    if kind == "fall":
        return -frac
    if kind == "arch":
        return math.sin(math.pi * frac)
    if kind == "wave":
        return math.sin(2 * math.pi * frac) * 0.7
    return 0.0


def _rhythm_for(tempo: int, beats_per_cycle: int, intensity: float,
                rng: random.Random) -> List[float]:
    if tempo < 70:
        bank = RHYTHM_BANK["slow"] + (RHYTHM_BANK["medium"] if intensity > 0.65 else [])
    elif tempo < 100:
        bank = RHYTHM_BANK["medium"] + (RHYTHM_BANK["fast"] if intensity > 0.7 else [])
    else:
        bank = RHYTHM_BANK["fast"] + RHYTHM_BANK["medium"]
    pattern = list(rng.choice(bank))
    total = sum(pattern)
    if abs(total - beats_per_cycle) > 1e-6:
        pattern = [p * beats_per_cycle / total for p in pattern]
    return pattern


def _phrase_tokens(raaga: Raaga, rng: random.Random, start: str, count: int,
                   contour: str, span: float, ornament: float,
                   cadence: Optional[str]) -> List[str]:
    tokens: List[str] = []
    cur = start
    start_deg = raaga.degree(cur)
    i = 0
    while i < count:
        remaining = count - i
        # Occasionally quote a characteristic phrase of the raaga verbatim.
        if (raaga.prayogas and remaining >= 3 and i > 0
                and rng.random() < 0.22 + 0.25 * ornament):
            frag = list(rng.choice(raaga.prayogas))
            _, octave = parse_swara(cur)
            frag_tokens = []
            for tok in frag[:remaining]:
                b, o = parse_swara(tok)
                frag_tokens.append(_with_octave(b, octave + o))
            tokens.extend(frag_tokens)
            cur = frag_tokens[-1]
            i += len(frag_tokens)
            continue

        frac = i / max(1, count - 1)
        target_deg = start_deg + _shape(contour, frac) * span
        cur_deg = raaga.degree(cur)
        if cur_deg < target_deg - 0.5:
            direction = 1
        elif cur_deg > target_deg + 0.5:
            direction = -1
        else:
            direction = rng.choice((1, -1, 1, -1, 0))
        if direction == 0:
            tokens.append(cur)
        else:
            steps = 1 if rng.random() < 0.75 else 2
            cur = raaga.step(cur, steps * direction, direction)
            tokens.append(cur)
        i += 1

    if cadence and tokens:
        tokens[-1] = cadence
    return tokens


def _section_register(raaga: Raaga, kind: SectionKind, tonic: int,
                      low: int, high: int) -> Tuple[int, int, float]:
    """Return (lo, hi, span) for the section's tessitura."""
    role = section_role(kind)
    if role == "hook":
        lo = tonic + 2
        hi = min(high, tonic + 14)
        span = 5.0
    elif role == "verse":
        lo = max(low, tonic - 5)
        hi = min(high, tonic + 9)
        span = 4.0
    else:  # instrumental
        lo = max(low, tonic - 7)
        hi = min(high, tonic + 16)
        span = 6.0
    return max(low, lo), max(lo + 5, hi), span


def generate_section_notes(raaga: Raaga, section: Section, opts: MelodyOptions,
                           seed: int, entry_token: Optional[str] = None
                           ) -> List[Note]:
    rng = random.Random(seed)
    cyc = cycle_seconds(opts.tempo_bpm, opts.beats_per_cycle)
    beat = beat_seconds(opts.tempo_bpm)
    lo, hi, span = _section_register(raaga, section.kind, opts.tonic_midi,
                                     opts.voice_low, opts.voice_high)
    intensity = section.intensity
    ornament = opts.ornament + (0.2 if section.kind.instrumental else 0.0)

    n_phrases = max(1, int(round(section.duration / cyc)))
    cur = entry_token or (rng.choice(raaga.graha) if raaga.graha else "S")
    cur = clamp_token(raaga, cur, opts.tonic_midi, lo, hi)

    notes: List[Note] = []
    t = section.start
    for p in range(n_phrases):
        if t >= section.end - 1e-3:
            break
        pattern = _rhythm_for(opts.tempo_bpm, opts.beats_per_cycle, intensity, rng)
        phrase_len = min(sum(pattern) * beat, section.end - t)
        if phrase_len <= 0.05:
            break
        last_phrase = (p == n_phrases - 1)
        if last_phrase:
            contour = "fall"
        elif section.kind in (SectionKind.PALLAVI, SectionKind.CHORUS):
            contour = "arch" if p % 2 == 0 else "rise"
        else:
            contour = rng.choice(CONTOURS)
        cadence = _nearest_nyasa(raaga, cur) if (last_phrase or rng.random() < 0.5) else None
        tokens = _phrase_tokens(raaga, rng, cur, len(pattern), contour,
                                span * (0.6 + 0.8 * intensity), ornament, cadence)

        # Breath: leave a small gap at the end of the phrase.
        breath = min(0.28, phrase_len * 0.12) if not section.kind.instrumental else \
            min(0.14, phrase_len * 0.06)
        usable = max(0.1, phrase_len - breath)
        durations = [d * beat for d in pattern]
        scale = usable / max(1e-6, sum(durations))
        durations = [d * scale for d in durations]

        for idx, (tok, dur) in enumerate(zip(tokens, durations)):
            tok = clamp_token(raaga, tok, opts.tonic_midi, lo, hi)
            midi = token_midi(raaga, tok, opts.tonic_midi)
            accent = 1.0 if idx == 0 else (0.9 if dur >= beat * 1.5 else 0.8)
            velocity = int(56 + 52 * intensity * accent)
            gamaka = ""
            if dur >= beat * 0.55 and rng.random() < 0.35 + 0.5 * ornament:
                gamaka = raaga.gamaka_for(tok)
            notes.append(Note(swara=tok, midi=midi, start=round(t, 4),
                              duration=round(max(0.06, dur), 4),
                              velocity=max(30, min(120, velocity)),
                              gamaka=gamaka, section_id=section.id))
            t += dur
            cur = tok
        t += breath
    return notes


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def generate(raaga: Raaga, opts: MelodyOptions,
             sections: Optional[List[Section]] = None,
             version: int = 1, label: str = "") -> MelodyVersion:
    sections = sections or plan_sections(opts.duration_target, opts.tempo_bpm,
                                         opts.beats_per_cycle, opts.song_type)
    melody = MelodyVersion(version=version, label=label or f"Tune v{version}",
                           raaga=raaga.name, tonic_midi=opts.tonic_midi,
                           tempo_bpm=opts.tempo_bpm,
                           beats_per_cycle=opts.beats_per_cycle,
                           sections=deepcopy(sections), seed=opts.seed)
    entry: Optional[str] = None
    for i, section in enumerate(melody.sections):
        notes = generate_section_notes(raaga, section, opts, opts.seed * 1000 + i, entry)
        melody.notes.extend(notes)
        if notes:
            entry = notes[-1].swara
    return melody


def regenerate_section(melody: MelodyVersion, raaga: Raaga, section_id: str,
                       opts: MelodyOptions, new_version: int,
                       label: str = "") -> MelodyVersion:
    """Replace one section's notes; every other note is carried over untouched."""
    section = melody.section_by_id(section_id)
    if section is None:
        raise KeyError(f"No such section: {section_id}")
    if section.locked:
        from ..core.versioning import LockedContentError
        raise LockedContentError(f"Section '{section.name}' is locked.")

    fresh = deepcopy(melody)
    fresh.version = new_version
    fresh.parent_version = melody.version
    fresh.state = ApprovalState.DRAFT
    fresh.audio_path = ""
    fresh.label = label or f"Tune v{new_version}"
    fresh.derived_from = f"v{melody.version}: regenerated {section.name}"
    fresh.notes = [n for n in fresh.notes if n.section_id != section_id]

    index = [s.id for s in melody.sections].index(section_id)
    prior = [n for n in melody.notes
             if n.section_id == (melody.sections[index - 1].id if index else "")]
    entry = prior[-1].swara if prior else None
    seed = opts.seed * 1000 + index + new_version * 7919
    fresh.notes.extend(generate_section_notes(raaga, fresh.sections[index], opts,
                                              seed, entry))
    fresh.notes.sort(key=lambda n: n.start)
    return fresh


def variation(melody: MelodyVersion, raaga: Raaga, opts: MelodyOptions,
              new_version: int, strength: float = 0.5,
              label: str = "") -> MelodyVersion:
    """A fresh take that keeps structure, tempo and every locked section."""
    fresh = deepcopy(melody)
    fresh.version = new_version
    fresh.parent_version = melody.version
    fresh.state = ApprovalState.DRAFT
    fresh.audio_path = ""
    fresh.label = label or f"Tune v{new_version}"
    fresh.derived_from = f"variation of v{melody.version}"
    fresh.seed = opts.seed

    kept = [n for n in melody.notes
            if melody.section_by_id(n.section_id)
            and melody.section_by_id(n.section_id).locked]
    fresh.notes = list(kept)
    entry = None
    for i, section in enumerate(fresh.sections):
        if section.locked:
            tail = [n for n in kept if n.section_id == section.id]
            entry = tail[-1].swara if tail else entry
            continue
        seed = opts.seed * 1000 + i + int(strength * 10007) + new_version * 104729
        notes = generate_section_notes(raaga, section, opts, seed, entry)
        fresh.notes.extend(notes)
        if notes:
            entry = notes[-1].swara
    fresh.notes.sort(key=lambda n: n.start)
    return fresh


def retempo(melody: MelodyVersion, new_bpm: int, new_version: int) -> MelodyVersion:
    """Change tempo without changing the tune."""
    factor = melody.tempo_bpm / max(1, new_bpm)
    fresh = deepcopy(melody)
    fresh.version = new_version
    fresh.parent_version = melody.version
    fresh.tempo_bpm = int(new_bpm)
    fresh.state = ApprovalState.DRAFT
    fresh.audio_path = ""
    fresh.label = f"Tune v{new_version}"
    fresh.derived_from = f"v{melody.version} at {new_bpm} bpm"
    for n in fresh.notes:
        n.start = round(n.start * factor, 4)
        n.duration = round(n.duration * factor, 4)
    for s in fresh.sections:
        s.start = round(s.start * factor, 4)
        s.end = round(s.end * factor, 4)
    return fresh


def transpose(melody: MelodyVersion, semitones: int, new_version: int) -> MelodyVersion:
    fresh = deepcopy(melody)
    fresh.version = new_version
    fresh.parent_version = melody.version
    fresh.tonic_midi = melody.tonic_midi + semitones
    fresh.state = ApprovalState.DRAFT
    fresh.audio_path = ""
    fresh.derived_from = f"v{melody.version} transposed {semitones:+d}"
    for n in fresh.notes:
        n.midi += semitones
    return fresh


def notes_in_range(melody: MelodyVersion, start: float, end: float) -> List[Note]:
    return [n for n in melody.notes if n.start < end and start < n.end]


def phrase_boundaries(melody: MelodyVersion, section_id: str = "",
                      gap: float = 0.12) -> List[List[int]]:
    """Group note indices into breath-separated phrases."""
    idx = [i for i, n in enumerate(melody.notes)
           if not section_id or n.section_id == section_id]
    phrases: List[List[int]] = []
    current: List[int] = []
    for pos, i in enumerate(idx):
        current.append(i)
        n = melody.notes[i]
        nxt = melody.notes[idx[pos + 1]] if pos + 1 < len(idx) else None
        if nxt is None or (nxt.start - n.end) >= gap:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    return phrases
