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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# The longest run a tune may share with a learned phrase before the
# originality checker (agent/originality.py, DEFAULT_MAX_RUN) rejects it.
# Kept equal by a test (tests/unit/test_idiom.py) rather than an import,
# because music/ must not import agent/.
MAX_QUOTE_NOTES = 6
# A phrase longer than that is quoted as a fragment of this many notes, the
# same limit practice puts on quoting an idiom (docs/DECISIONS.md,
# "Originality is enforced"): a six-note quote passes the checker but scores
# a third on originality, a three-note one keeps most of it.
QUOTE_FRAGMENT_NOTES = 3


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
    # What the agent's lessons say before this attempt (agent/guidance.py's
    # ``Guidance``, applied here the same way agent/practice.py applies it).
    # Typed ``Any``: music/ must not import agent/.  The object only needs to
    # duck-type ``Guidance`` (``is_empty``, ``allows_transition``,
    # ``allows_ending``, ``replays``, and the plain attributes read below).
    # With ``None`` (or an empty ``Guidance``) the draw sequence is
    # byte-identical to a build with no guidance support at all.
    guidance: Optional[Any] = None


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


def enforce_direction(raaga: Raaga, notes: List[Note], tonic: int) -> int:
    """Make every move legal in the direction it actually travels.

    The generator walks in scale-degree space, taking each next note from the
    ascending or descending ladder as the phrase requires.  Two things then
    happen to that walk which know nothing about direction: octaves are
    placed to fit a register, and phrases are joined into a section.  Either
    can turn a legal descending step into an ascending leap - the note is
    unchanged, the motion is reversed - and in a raaga whose arohanam and
    avarohanam differ, rising onto a descent-only note is wrong.  Measured on
    the library's four asymmetric raagas, that happened on 4 to 42 moves in
    735, in as many as 16 of 20 seeds.

    In a raaga whose two ladders hold the same swaras no motion can be
    illegal, so nothing is touched and the line is bit-for-bit what the walk
    produced.  That is also why this went unnoticed: everything was measured
    on Keeravani, which is one of those.

    A note that breaks its move is replaced by the nearest pitch that does
    not, keeping the direction of travel.  Nothing further than a fifth away
    is substituted - past that the repair would be a worse artefact than the
    fault - and the evaluator still reports what is left.  Returns the number
    of notes changed.
    """
    if not notes:
        return 0
    ascending = set(raaga.ascending)
    descending = set(raaga.descending)
    if ascending == descending:
        return 0

    repaired = 0
    for i in range(1, len(notes)):
        previous, current = notes[i - 1], notes[i]
        if current.midi == previous.midi:
            continue
        rising = current.midi > previous.midi
        allowed = ascending if rising else descending
        if parse_swara(current.swara)[0] in allowed:
            continue

        best, best_gap = None, None
        for swara in sorted(allowed):
            for octave in range(-3, 4):
                token = _with_octave(swara, octave)
                midi = token_midi(raaga, token, tonic)
                if midi == previous.midi or (midi > previous.midi) != rising:
                    continue
                gap = abs(midi - current.midi)
                if best_gap is None or gap < best_gap:
                    best, best_gap = (token, midi), gap
        if best is None or best_gap > 7:
            continue
        current.swara, current.midi = best[0], best[1]
        repaired += 1
    return repaired


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


def _guidance_or_none(guidance: Optional[Any]) -> Optional[Any]:
    """``None`` unless ``guidance`` is present and has something to say -
    the gate every guided branch below checks before it makes a decision
    that an unguided run would not make."""
    if guidance is None:
        return None
    if hasattr(guidance, "is_empty") and guidance.is_empty():
        return None
    return guidance


def _prayoga_source(raaga: Raaga, frag: Sequence[str]) -> Tuple[str, str, str]:
    """(phrase_id, origin, source) for a quoted fragment.  Matched first as a
    whole phrase in ``raaga.prayoga_sources`` (agent/learned.py fills this
    for heard phrases), then as a contiguous run inside a longer heard
    phrase - a fragment cut from a phrase longer than ``MAX_QUOTE_NOTES``
    still traces back to what it was cut from.  A fragment matching neither
    is still real provenance, attributed to the raaga library."""
    sources = getattr(raaga, "prayoga_sources", None) or {}
    frag = list(frag)
    joined = " ".join(frag)
    info = sources.get(joined)
    if info is not None:
        return (info.get("phrase_id", ""),
               info.get("origin") or "the raaga library", "learned")
    n = len(frag)
    if n:
        for key, info in sources.items():
            parts = key.split(" ")
            for start in range(0, len(parts) - n + 1):
                if parts[start:start + n] == frag:
                    return (info.get("phrase_id", ""),
                           info.get("origin") or "the raaga library", "learned")
    return "", "the raaga library", "prayoga"


def _cadence_with_guidance(raaga: Raaga, guidance: Any, cur: str,
                           cadence: Optional[str]) -> Optional[str]:
    """Steer which resting note a phrase lands on: never one of
    ``avoid_endings``, forced when ``must_end_on_nyasa`` even where the coin
    flip found no cadence, and preferring a note that is both jeeva and
    nyasa when ``prefer_jeeva`` asks for one - else the nearest nyasa."""
    must_end = bool(getattr(guidance, "must_end_on_nyasa", False))
    if cadence is None and not must_end:
        return cadence
    avoid = set(getattr(guidance, "avoid_endings", None) or ())
    if cadence is not None and parse_swara(cadence)[0] not in avoid:
        return cadence
    pool = [n for n in raaga.nyasa if parse_swara(n)[0] not in avoid] or list(raaga.nyasa)
    if not pool:
        return cadence
    if getattr(guidance, "prefer_jeeva", False):
        jeeva_set = set(raaga.jeeva)
        jeeva_nyasa = [n for n in pool if parse_swara(n)[0] in jeeva_set]
        if jeeva_nyasa:
            pool = jeeva_nyasa
    _, cur_octave = parse_swara(cur)
    cur_deg = raaga.degree(cur)
    best = min(pool, key=lambda n: abs(
        raaga.degree(_with_octave(parse_swara(n)[0], cur_octave)) - cur_deg))
    return _with_octave(parse_swara(best)[0], cur_octave)


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
                   cadence: Optional[str],
                   guidance: Optional[Any] = None,
                   quotes: Optional[List[Tuple[int, List[str]]]] = None
                   ) -> List[str]:
    idiom = getattr(raaga, "idiom", None)
    g = _guidance_or_none(guidance)
    tokens: List[str] = []
    cur = start
    start_deg = raaga.degree(cur)
    i = 0
    while i < count:
        remaining = count - i
        # Occasionally quote a characteristic phrase of the raaga verbatim -
        # or, when it runs longer than the originality checker would allow,
        # as a fragment transposed into the current octave.  ``quote_more``
        # raises the odds of this branch firing at all.
        quote_bias = float(g.quote_more) if g is not None else 0.0
        if (raaga.prayogas and remaining >= 3 and i > 0
                and rng.random() < 0.22 + 0.25 * ornament + quote_bias):
            frag = list(rng.choice(raaga.prayogas))
            # A phrase longer than MAX_QUOTE_NOTES is quoted as a fragment
            # of QUOTE_FRAGMENT_NOTES, never whole; a deterministic window
            # (no rng draw) picks which part.  No library prayoga exceeds
            # MAX_QUOTE_NOTES (max 5, verified), so this never fires for an
            # unstudied raaga and changes nothing there.
            if len(frag) > MAX_QUOTE_NOTES:
                start_at = ((i + len(tokens))
                           % (len(frag) - QUOTE_FRAGMENT_NOTES + 1))
                frag = frag[start_at:start_at + QUOTE_FRAGMENT_NOTES]
            frag = frag[:remaining]
            _, octave = parse_swara(cur)
            frag_tokens = []
            for tok in frag:
                b, o = parse_swara(tok)
                frag_tokens.append(_with_octave(b, octave + o))
            # A fragment that would replay a run this line was told to
            # avoid is dropped - but the draws above already happened, so a
            # guided attempt makes exactly the same rng calls as an
            # unguided one and just falls through to the walk below.
            blocked = g is not None and g.replays(tokens + frag_tokens)
            if not blocked:
                if quotes is not None:
                    quotes.append((i, list(frag)))
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
        elif idiom is not None:
            direction = idiom.pick_direction(cur, rng)
        else:
            direction = rng.choice((1, -1, 1, -1, 0))
        if direction == 0:
            tokens.append(cur)
        else:
            if idiom is not None:
                steps = idiom.pick_steps(cur, direction, rng)
            else:
                step_bias = float(g.prefer_step) if g is not None else 0.0
                steps = 1 if rng.random() < min(0.98, 0.75 + step_bias) else 2
            nxt = raaga.step(cur, steps * direction, direction)
            if g is not None and not g.allows_transition(cur, nxt):
                alt = raaga.step(cur, -steps * direction, -direction)
                if g.allows_transition(cur, alt):
                    nxt = alt
            if (g is not None and g.vary_more and tokens
                    and sum(1 for t in tokens
                           if parse_swara(t)[0] == parse_swara(nxt)[0]) >= 2):
                alt = raaga.step(cur, -steps * direction, -direction)
                if g.allows_transition(cur, alt) and \
                        parse_swara(alt)[0] != parse_swara(nxt)[0]:
                    nxt = alt
            if g is not None and g.replays(tokens + [nxt]):
                alt = raaga.step(cur, -steps * direction, -direction)
                if g.allows_transition(cur, alt) and not g.replays(tokens + [alt]):
                    nxt = alt
            cur = nxt
            tokens.append(cur)
        i += 1

    if cadence and tokens:
        # The cadence overwrites the phrase's last token even when that
        # token came from a quote; shrink (or drop) the recorded quote so
        # provenance never claims a swara the cadence just replaced.
        if quotes:
            offset, frag = quotes[-1]
            if offset + len(frag) - 1 == len(tokens) - 1:
                quotes[-1] = (offset, frag[:-1]) if len(frag) > 1 else None
                if quotes[-1] is None:
                    quotes.pop()
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
                           seed: int, entry_token: Optional[str] = None,
                           provenance: Optional[List[dict]] = None
                           ) -> List[Note]:
    rng = random.Random(seed)
    cyc = cycle_seconds(opts.tempo_bpm, opts.beats_per_cycle)
    beat = beat_seconds(opts.tempo_bpm)
    lo, hi, span = _section_register(raaga, section.kind, opts.tonic_midi,
                                     opts.voice_low, opts.voice_high)
    intensity = section.intensity
    ornament = opts.ornament + (0.2 if section.kind.instrumental else 0.0)
    guidance = _guidance_or_none(getattr(opts, "guidance", None))

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
        if last_phrase or rng.random() < 0.5:
            idiom = getattr(raaga, "idiom", None)
            cadence = (idiom.cadence_for(raaga, cur) if idiom is not None
                      else _nearest_nyasa(raaga, cur))
        else:
            cadence = None
        if guidance is not None:
            cadence = _cadence_with_guidance(raaga, guidance, cur, cadence)
        quotes_acc: Optional[List[Tuple[int, List[str]]]] = \
            [] if provenance is not None else None
        tokens = _phrase_tokens(raaga, rng, cur, len(pattern), contour,
                                span * (0.6 + 0.8 * intensity), ornament, cadence,
                                guidance=guidance, quotes=quotes_acc)

        # Breath: leave a small gap at the end of the phrase.
        breath = min(0.28, phrase_len * 0.12) if not section.kind.instrumental else \
            min(0.14, phrase_len * 0.06)
        usable = max(0.1, phrase_len - breath)
        durations = [d * beat for d in pattern]
        scale = usable / max(1e-6, sum(durations))
        durations = [d * scale for d in durations]

        phrase_note_offset = len(notes)
        for idx, (tok, dur) in enumerate(zip(tokens, durations)):
            tok = clamp_token(raaga, tok, opts.tonic_midi, lo, hi)
            midi = token_midi(raaga, tok, opts.tonic_midi)
            accent = 1.0 if idx == 0 else (0.9 if dur >= beat * 1.5 else 0.8)
            velocity = int(56 + 52 * intensity * accent)
            gamaka = ""
            if dur >= beat * 0.55 and rng.random() < 0.35 + 0.5 * ornament:
                gamaka = raaga.gamaka_for(tok)
            if guidance is not None and guidance.add_gamaka and not gamaka \
                    and dur >= beat * 0.5:
                gamaka = raaga.gamaka_for(tok) or "kampita"
            notes.append(Note(swara=tok, midi=midi, start=round(t, 4),
                              duration=round(max(0.06, dur), 4),
                              velocity=max(30, min(120, velocity)),
                              gamaka=gamaka, section_id=section.id))
            t += dur
            cur = tok
        t += breath

        if quotes_acc:
            for token_offset, frag in quotes_acc:
                start_idx = phrase_note_offset + token_offset
                end_idx = min(start_idx + len(frag) - 1, len(notes) - 1)
                if start_idx > end_idx or start_idx < 0:
                    continue
                phrase_id, origin, source = _prayoga_source(raaga, frag)
                provenance.append({
                    "start": start_idx, "end": end_idx,
                    "swaras": " ".join(frag), "source": source,
                    "phrase_id": phrase_id, "origin": origin,
                    "section_id": section.id,
                })
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
        local_provenance: List[dict] = []
        notes = generate_section_notes(raaga, section, opts, opts.seed * 1000 + i,
                                       entry, provenance=local_provenance)
        offset = len(melody.notes)
        melody.notes.extend(notes)
        for entry_dict in local_provenance:
            melody.provenance.append({**entry_dict,
                                     "start": entry_dict["start"] + offset,
                                     "end": entry_dict["end"] + offset})
        if notes:
            entry = notes[-1].swara
    # Last, on the whole line: the direction of a move is only finally known
    # once octaves are placed and the sections are joined, and it is the
    # direction that decides whether a note is allowed (see enforce_direction).
    enforce_direction(raaga, melody.notes, opts.tonic_midi)
    return melody


def _copy_provenance_for_kept(old_melody: MelodyVersion, section_id: str) -> List[dict]:
    """The old melody's provenance for one untouched section, reindexed to
    be local (0-based within that section's own notes) so a caller can shift
    it onto wherever the section lands in the rebuilt note list."""
    idxs = [i for i, n in enumerate(old_melody.notes) if n.section_id == section_id]
    if not idxs:
        return []
    lo = idxs[0]
    return [{**e, "start": e["start"] - lo, "end": e["end"] - lo}
            for e in old_melody.provenance if e.get("section_id") == section_id]


def regenerate_section(melody: MelodyVersion, raaga: Raaga, section_id: str,
                       opts: MelodyOptions, new_version: int,
                       label: str = "") -> MelodyVersion:
    """Replace one section's notes; every other note - and its provenance -
    is carried over untouched."""
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

    index = [s.id for s in melody.sections].index(section_id)
    prior = [n for n in melody.notes
             if n.section_id == (melody.sections[index - 1].id if index else "")]
    entry = prior[-1].swara if prior else None
    seed = opts.seed * 1000 + index + new_version * 7919
    local_provenance: List[dict] = []
    new_notes = generate_section_notes(raaga, fresh.sections[index], opts,
                                       seed, entry, provenance=local_provenance)

    # Rebuilt section by section, in time order (sections never overlap), so
    # this reproduces exactly what "carry every other note, sort by start"
    # produced before, while keeping provenance indices honest.
    rebuilt_notes: List[Note] = []
    rebuilt_provenance: List[dict] = []
    for s in fresh.sections:
        offset = len(rebuilt_notes)
        if s.id == section_id:
            rebuilt_notes.extend(new_notes)
            for e in local_provenance:
                rebuilt_provenance.append({**e, "start": e["start"] + offset,
                                          "end": e["end"] + offset})
        else:
            kept_notes = [n for n in melody.notes if n.section_id == s.id]
            rebuilt_notes.extend(kept_notes)
            for e in _copy_provenance_for_kept(melody, s.id):
                rebuilt_provenance.append({**e, "start": e["start"] + offset,
                                          "end": e["end"] + offset})
    fresh.notes = rebuilt_notes
    fresh.provenance = rebuilt_provenance
    return fresh


def variation(melody: MelodyVersion, raaga: Raaga, opts: MelodyOptions,
              new_version: int, strength: float = 0.5,
              label: str = "") -> MelodyVersion:
    """A fresh take that keeps structure, tempo, every locked section and its
    provenance."""
    fresh = deepcopy(melody)
    fresh.version = new_version
    fresh.parent_version = melody.version
    fresh.state = ApprovalState.DRAFT
    fresh.audio_path = ""
    fresh.label = label or f"Tune v{new_version}"
    fresh.derived_from = f"variation of v{melody.version}"
    fresh.seed = opts.seed

    rebuilt_notes: List[Note] = []
    rebuilt_provenance: List[dict] = []
    entry = None
    for i, section in enumerate(fresh.sections):
        offset = len(rebuilt_notes)
        if section.locked:
            kept_notes = [n for n in melody.notes if n.section_id == section.id]
            rebuilt_notes.extend(kept_notes)
            for e in _copy_provenance_for_kept(melody, section.id):
                rebuilt_provenance.append({**e, "start": e["start"] + offset,
                                          "end": e["end"] + offset})
            if kept_notes:
                entry = kept_notes[-1].swara
            continue
        seed = opts.seed * 1000 + i + int(strength * 10007) + new_version * 104729
        local_provenance: List[dict] = []
        notes = generate_section_notes(raaga, section, opts, seed, entry,
                                       provenance=local_provenance)
        rebuilt_notes.extend(notes)
        for e in local_provenance:
            rebuilt_provenance.append({**e, "start": e["start"] + offset,
                                      "end": e["end"] + offset})
        if notes:
            entry = notes[-1].swara
    fresh.notes = rebuilt_notes
    fresh.provenance = rebuilt_provenance
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
