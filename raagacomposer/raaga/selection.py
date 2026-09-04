"""Raaga selection engine (spec section 4 step 2).

Turns a free-form creative brief -- "lonely, late at night, but still warm" --
into ranked raaga suggestions with a stated reason.  The creator accepts,
rejects, asks for alternatives or overrides; nothing is chosen silently.

A language model may be consulted when one is configured, but the ranking below
always runs so the app behaves identically with no credentials at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.models import CreativeBrief
from . import emotion
from .library import Raaga, RaagaLibrary, library

# Free-text feel word -> mood tokens used by the raaga data.
FEEL_LEXICON: Dict[str, Sequence[str]] = {
    "romantic": ("romantic", "warm", "intimate"),
    "love": ("romantic", "warm"),
    "longing": ("longing", "yearning", "separation"),
    "yearn": ("longing", "yearning"),
    "miss": ("longing", "separation", "lonely"),
    "sad": ("sad", "pathos", "melancholy"),
    "sorrow": ("sad", "pathos", "grief"),
    "grief": ("grief", "sad", "heavy"),
    "cry": ("sad", "pathos"),
    "tear": ("sad", "pathos"),
    "lonely": ("lonely", "sad", "pensive"),
    "alone": ("lonely", "pensive"),
    "night": ("night", "late night", "still"),
    "midnight": ("night", "late night", "mystical"),
    "dawn": ("dawn", "morning", "calm"),
    "morning": ("morning", "serene", "calm"),
    "evening": ("evening", "warm", "reflective"),
    "warm": ("warm", "romantic", "tender"),
    "tender": ("tender", "gentle", "intimate"),
    "gentle": ("gentle", "calm", "soothing"),
    "soft": ("gentle", "calm", "intimate"),
    "happy": ("joyful", "bright", "light"),
    "joy": ("joyful", "celebration", "bright"),
    "celebration": ("celebration", "festive", "auspicious"),
    "wedding": ("auspicious", "celebration", "festive"),
    "festival": ("festive", "celebration", "energetic"),
    "dance": ("energetic", "festive", "bright"),
    "energy": ("energetic", "bright"),
    "energetic": ("energetic", "festive"),
    "fast": ("energetic", "festive"),
    "devotional": ("devotional", "prayerful", "serene"),
    "prayer": ("prayerful", "devotional", "calm"),
    "temple": ("devotional", "prayerful", "dignified"),
    "god": ("devotional", "prayerful"),
    "heroic": ("heroic", "majestic", "grand"),
    "brave": ("heroic", "grand"),
    "battle": ("heroic", "intense", "grand"),
    "aggressive": ("intense", "heavy", "brooding"),
    "angry": ("intense", "brooding", "heavy"),
    "dark": ("dark", "brooding", "night"),
    "mystical": ("mystical", "meditative", "still"),
    "meditative": ("meditative", "contemplative", "calm"),
    "calm": ("calm", "soothing", "serene"),
    "peaceful": ("calm", "serene", "soothing"),
    "nostalgia": ("nostalgia", "wistful", "reflective"),
    "memory": ("nostalgia", "reflective", "wistful"),
    "childhood": ("innocent", "pastoral", "light"),
    "village": ("pastoral", "folk", "earthy"),
    "folk": ("folk", "earthy"),
    "rain": ("pastoral", "reflective", "tender"),
    "separation": ("separation", "longing", "sad"),
    "hope": ("hopeful", "bright"),
    "hopeful": ("hopeful", "bright", "warm"),
    "grand": ("grand", "majestic"),
    "majestic": ("majestic", "grand", "dignified"),
    "intimate": ("intimate", "tender", "gentle"),
    "closing": ("closing", "resolution", "auspicious"),
    "opening": ("invocation", "auspicious", "bright"),
    "cinematic": ("cinematic", "narrative", "grand"),
    "bittersweet": ("bittersweet", "wistful", "nostalgia"),
}

TEMPO_HINTS = {
    "slow": 60, "very slow": 48, "medium": 84, "moderate": 84,
    "fast": 112, "very fast": 132, "upbeat": 108, "gentle": 66,
}


@dataclass
class RaagaSuggestion:
    name: str
    score: float
    rationale: str
    raaga: Optional[Raaga] = None
    matched_moods: List[str] = field(default_factory=list)

    def short(self) -> str:
        return f"{self.name} - {self.rationale}"


def expand_feel_words(*texts: str) -> List[str]:
    """Extract mood tokens from any amount of free-form creator language."""
    words: List[str] = []
    blob = " ".join(t.lower() for t in texts if t)
    for token in re.findall(r"[a-z]+", blob):
        for key, moods in FEEL_LEXICON.items():
            if token.startswith(key[:4]) and (key in token or token in key):
                words.extend(moods)
                break
    for key, moods in FEEL_LEXICON.items():
        if " " in key and key in blob:
            words.extend(moods)
    seen: List[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return seen


def infer_tempo(brief: CreativeBrief, raaga: Optional[Raaga]) -> int:
    if brief.tempo_preference:
        return int(brief.tempo_preference)
    blob = " ".join((brief.feel, brief.mood, brief.notes, brief.situation)).lower()
    for key, bpm in sorted(TEMPO_HINTS.items(), key=lambda kv: -len(kv[0])):
        if key in blob:
            return bpm
    words = expand_feel_words(blob)
    energetic = {"energetic", "festive", "celebration", "bright", "joyful"}
    slow = {"sad", "grief", "lonely", "meditative", "still", "pathos", "prayerful"}
    if raaga and raaga.tempo_range:
        lo, hi = raaga.tempo_range[0], raaga.tempo_range[-1]
    else:
        lo, hi = 56, 108
    if any(w in energetic for w in words):
        return int(lo + 0.75 * (hi - lo))
    if any(w in slow for w in words):
        return int(lo + 0.15 * (hi - lo))
    return int((lo + hi) / 2)


def suggest(brief: CreativeBrief, lib: Optional[RaagaLibrary] = None,
            limit: int = 4) -> List[RaagaSuggestion]:
    lib = lib or library()

    # An explicit request always wins and is reported as such.
    explicit = None
    if brief.raaga_preference:
        explicit = lib.get(brief.raaga_preference)
    if explicit is None:
        explicit = lib.find_in_text(" ".join((brief.feel, brief.notes, brief.situation)))

    # The Stage 1 pack's engine (raaga/emotion.py) does the ranking: the
    # brief becomes a fourteen-dimension emotion target, every raaga is
    # scored on how its blocks and curated character match it, the pack's
    # contradiction penalties and block bonuses adjust that, and the five
    # returned are a spread rather than five variations on one profile.
    graded = emotion.rank(brief, lib.all(), limit=limit)
    scored = [
        RaagaSuggestion(name=item.name, score=item.score, raaga=item.raaga,
                        matched_moods=list(item.tags),
                        rationale=_rationale(item, item.raaga is explicit))
        for item in graded
    ]

    if explicit is not None and not any(s.raaga is explicit for s in scored):
        # Being asked for by name outranks anything the engine has to say,
        # and the list is still a list, so it goes at the front rather than
        # replacing what was suggested.
        scored.insert(0, RaagaSuggestion(
            name=explicit.name, score=100.0, raaga=explicit,
            rationale=_rationale(None, True, explicit)))
        scored = scored[:limit]
    elif explicit is not None:
        scored.sort(key=lambda s: (s.raaga is not explicit, -s.score, s.name))

    if not scored:
        # A brief that says nothing at all: answer with something curated
        # rather than a parent scale nobody could give a reason for.
        fallback = lib.get("Mohanam") or next(
            (r for r in lib.all() if r.moods), lib.all()[0])
        scored = [RaagaSuggestion(
            name=fallback.name, score=10.0, raaga=fallback,
            rationale="A safe, open-sounding default while the brief is still thin.")]

    return scored[:limit]


def _rationale(scored: Optional["emotion.Scored"], explicit: bool,
               raaga: Optional[Raaga] = None) -> str:
    """One sentence, from the block logic (pack document 05 section 4)."""
    raaga = raaga or (scored.raaga if scored else None)
    if explicit:
        return f"You asked for {raaga.name}. {raaga.character()}".strip()
    if scored is None:
        return (raaga.character() if raaga else "") or "Fits the general shape."
    head = f"{emotion.sentence_case(scored.role)}: " if scored.role else ""
    return (head + scored.reason).strip()


def compare(a: Raaga, b: Raaga) -> str:
    """Human-readable comparison for the raaga panel."""
    lines = [f"{a.name}  vs  {b.name}", ""]
    lines.append(f"Arohanam    {' '.join(a.arohanam)}")
    lines.append(f"            {' '.join(b.arohanam)}")
    lines.append(f"Avarohanam  {' '.join(a.avarohanam)}")
    lines.append(f"            {' '.join(b.avarohanam)}")
    lines.append("")
    only_a = [s for s in a.allowed if s not in b.allowed]
    only_b = [s for s in b.allowed if s not in a.allowed]
    lines.append(f"Only in {a.name}: {', '.join(only_a) or '-'}")
    lines.append(f"Only in {b.name}: {', '.join(only_b) or '-'}")
    lines.append("")
    lines.append(f"{a.name} moods: {', '.join(a.moods)}")
    lines.append(f"{b.name} moods: {', '.join(b.moods)}")
    lines.append("")
    lines.append(f"{a.name}: {a.notes}")
    lines.append(f"{b.name}: {b.notes}")
    return "\n".join(lines)
