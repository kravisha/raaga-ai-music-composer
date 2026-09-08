"""Lyrics generator (spec sections 4 step 4, 12.15).

The generator writes *to the tune*: every line is built to land exactly on the
syllable count and stress pattern of a phrase slot produced by
:mod:`raagacomposer.lyrics.fitting`.

Two engines sit behind one call.  When a language model provider is configured
it is asked for lines with the required syllable counts and the result is
checked and re-fitted.  With no credentials the local lexicon engine runs, so
the workflow is never blocked -- lines are simple and thematic rather than
poetry, and the creator edits any line without touching the locked tune.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.models import CreativeBrief, LyricsVersion, MelodyVersion
from .fitting import (PhraseSlot, build_slots, count_syllables, fit_lines,
                      syllabify)

log = get_logger("lyrics")

# Transliterated word pools by theme.  Kept small and singable on purpose:
# open vowels, no consonant clusters that fight a sustained note.
POOLS: Dict[str, Dict[str, List[str]]] = {
    "tamil": {
        "romantic": ["kaadhal", "nesam", "aasai", "unnai", "ennai", "iniya",
                     "kaiyil", "punnagai", "idhayam"],
        "sad": ["thanimai", "kanneer", "pirivu", "vedhanai", "thavikkiren",
                "maunam", "izhandhen"],
        "longing": ["thedal", "meendum", "ninaivu", "thooram", "kaathiruppen",
                    "varuvaaya", "yaeno"],
        "night": ["iravu", "nilavu", "nizhal", "vaanam", "vidiyal", "kanavu"],
        "nature": ["mazhai", "kaatru", "kadal", "alai", "pookkal", "vasantham",
                   "karaiyil", "oli"],
        "devotional": ["deivam", "arul", "paadhangal", "nambikkai", "sharanam"],
        "celebration": ["kondaadu", "magizhchi", "thiruvizha", "aadungal", "isai"],
        "connective": ["endrum", "innum", "yaaro", "adhu", "idhu", "vaa", "nee",
                       "naan", "oru", "en"],
    },
    "hindi": {
        "romantic": ["pyaar", "chahat", "tumhe", "mujhe", "dhadkan", "muskaan",
                     "haathon", "dil"],
        "sad": ["tanha", "aansu", "judaai", "khamoshi", "tootha", "gham"],
        "longing": ["talash", "phir", "yaad", "doori", "intezaar", "aaoge", "kyun"],
        "night": ["raat", "chandni", "saaya", "aasman", "subah", "sapna"],
        "nature": ["baarish", "hawa", "samundar", "lehar", "phool", "bahaar",
                   "kinara", "roshni"],
        "devotional": ["bhagwan", "kripa", "charan", "vishwas", "sharan"],
        "celebration": ["jashn", "khushi", "tyohaar", "naacho", "sangeet"],
        "connective": ["hamesha", "abhi", "koi", "woh", "yeh", "aa", "tum",
                       "main", "ek", "mera"],
    },
    "telugu": {
        "romantic": ["prema", "isthamu", "ninnu", "nannu", "gundhe", "chirunavvu",
                     "chethilo"],
        "sad": ["ontari", "kanneeru", "edabaatu", "mounam", "badha"],
        "longing": ["vethuku", "marala", "gnapakam", "dooram", "eduruchoosthu",
                    "vasthava", "enduko"],
        "night": ["ratri", "vennela", "nidra", "aakasam", "tellavaru", "kala"],
        "nature": ["vaana", "gaali", "samudram", "alalu", "puvvulu", "vasantham",
                   "teeram", "velugu"],
        "devotional": ["devudu", "krupa", "paadalu", "nammakam", "sharanu"],
        "celebration": ["sambaram", "santhosham", "panduga", "aadandi", "sangeetham"],
        "connective": ["eppudu", "inka", "evaro", "adi", "idi", "raa", "nuvvu",
                       "nenu", "oka", "naa"],
    },
    "english": {
        "romantic": ["love", "closer", "hold you", "gently", "yours", "smile",
                     "heartbeat"],
        "sad": ["lonely", "tears", "parting", "silence", "broken", "sorrow"],
        "longing": ["searching", "again", "memory", "distance", "waiting",
                    "will you", "why"],
        "night": ["midnight", "moonlight", "shadow", "the sky", "morning", "dream"],
        "nature": ["rainfall", "the wind", "the ocean", "waves", "flowers",
                   "springtime", "shoreline", "light"],
        "devotional": ["grace", "mercy", "praying", "faithful", "surrender"],
        "celebration": ["dancing", "joyful", "festival", "sing out", "music"],
        "connective": ["always", "still", "someone", "that", "this", "come",
                       "you", "I", "one", "my"],
    },
}

MOOD_TO_THEMES: Dict[str, List[str]] = {
    "romantic": ["romantic", "night", "nature"],
    "love": ["romantic", "nature"],
    "longing": ["longing", "sad", "night"],
    "sad": ["sad", "longing", "night"],
    "lonely": ["sad", "night", "longing"],
    "devotional": ["devotional", "nature"],
    "celebration": ["celebration", "nature"],
    "festive": ["celebration"],
    "energetic": ["celebration", "nature"],
    "night": ["night", "sad"],
    "hopeful": ["nature", "romantic"],
    "upbeat": ["celebration", "nature"],
    "excited": ["celebration", "nature"],
    "nostalgia": ["longing", "night"],
}


def _language_pool(language: str) -> Dict[str, List[str]]:
    key = (language or "english").strip().lower()
    for name, pool in POOLS.items():
        if name in key or key in name:
            return pool
    return POOLS["english"]


def _themes_for(brief: CreativeBrief) -> List[str]:
    blob = " ".join((brief.mood, brief.feel, brief.situation, brief.notes)).lower()
    themes: List[str] = []
    for mood, group in MOOD_TO_THEMES.items():
        if mood in blob:
            themes.extend(group)
    if not themes:
        themes = ["romantic", "nature", "night"]
    themes.append("connective")
    seen: List[str] = []
    for t in themes:
        if t not in seen:
            seen.append(t)
    return seen


def _syl(word: str) -> int:
    return sum(len(syllabify(w)) for w in word.split()) or 1


def _final_vowel(text: str) -> str:
    m = re.findall(r"(aa|ee|oo|ai|au|[aeiou])", text.lower())
    return m[-1] if m else ""


def make_line(target: int, words: Sequence[str], rng: random.Random,
              rhyme_with: str = "") -> str:
    """Assemble a line with exactly ``target`` syllables."""
    if target <= 0:
        return ""
    counts: Dict[int, List[str]] = {}
    for w in words:
        counts.setdefault(_syl(w), []).append(w)
    if not counts:
        return "la " * target

    chosen: List[str] = []
    remaining = target
    guard = 0
    while remaining > 0 and guard < 40:
        guard += 1
        options = [n for n in counts if n <= remaining]
        if not options:
            break
        if remaining in counts:
            pool = list(counts[remaining])
            if rhyme_with:
                rhyming = [w for w in pool
                           if _final_vowel(w) == _final_vowel(rhyme_with)]
                if rhyming:
                    pool = rhyming
            chosen.append(rng.choice(pool))
            remaining = 0
            break
        n = rng.choice(sorted(options, reverse=True)[:3] or options)
        candidates = [w for w in counts[n] if w not in chosen] or counts[n]
        chosen.append(rng.choice(candidates))
        remaining -= n
    while remaining > 0:
        chosen.append("aa" if remaining == 1 else "aa aa")
        remaining -= 1 if remaining == 1 else 2
    return " ".join(chosen)


def generate_lines(slots: Sequence[PhraseSlot], brief: CreativeBrief,
                   seed: int = 3) -> List[str]:
    rng = random.Random(seed)
    pool = _language_pool(brief.language)
    themes = _themes_for(brief)
    words: List[str] = []
    for t in themes:
        words.extend(pool.get(t, []))
    if not words:
        words = sum(pool.values(), [])

    from .fitting import count_syllables

    # Index each phrase within its own section, so a repeated pallavi can reuse
    # the *matching* phrase rather than repeating line one over and over.
    phrase_index: List[int] = []
    seen_sections: Dict[str, int] = {}
    for slot in slots:
        n = seen_sections.get(slot.section_id, 0)
        phrase_index.append(n)
        seen_sections[slot.section_id] = n + 1

    lines: List[str] = []
    hook: Dict[tuple, str] = {}
    for i, slot in enumerate(slots):
        stem = re.sub(r"\s*\d+$", "", slot.section_name).strip().lower()
        key = (stem, phrase_index[i])
        earlier = hook.get(key)
        if earlier and count_syllables(earlier) == slot.syllable_count:
            lines.append(earlier)
            continue
        rhyme = lines[-1] if (i % 2 == 1 and lines) else ""
        recent = set(" ".join(lines[-2:]).split())
        line = make_line(slot.syllable_count,
                         [w for w in words if w not in recent] or words,
                         rng, rhyme_with=rhyme)
        lines.append(line)
        hook[key] = line
    return lines


def generate(melody: MelodyVersion, brief: CreativeBrief, version: int = 1,
             seed: int = 3, llm=None,
             previous: Optional[LyricsVersion] = None,
             section_ids: Optional[Sequence[str]] = None) -> LyricsVersion:
    """Produce a lyrics version fitted to *melody*.

    ``section_ids`` writes words for those sections only.  Every other
    phrase keeps the words it already had, so a creator who has settled the
    Pallavi and wants the Charanam rewritten does not lose the Pallavi to
    get it.  Passing none means the whole song, which is what every existing
    caller asks for.
    """
    slots = build_slots(melody)
    if not slots:
        return LyricsVersion(version=version, language=brief.language,
                             melody_version=melody.version,
                             notes="The tune has no vocal phrases yet.")

    chosen = set(section_ids or ())
    targets = [i for i, slot in enumerate(slots)
               if not chosen or slot.section_id in chosen]
    if chosen and not targets:
        # Asked for sections that have no sung phrases: say so rather than
        # quietly rewriting the whole song, which is the one outcome the
        # selection was meant to prevent.
        raise ValueError("The chosen sections have no phrases to sing.")
    wanted = [slots[i] for i in targets]

    lines: List[str] = []
    writer = ""
    if llm is not None and getattr(llm, "available", False):
        try:
            lines = [str(t) if t is not None else "" for t in llm.write_lyrics(wanted, brief)]
            writer = f"llm:{getattr(llm, 'name', 'llm')}"
            log.info("lyrics drafted by %s", getattr(llm, "name", "llm"))
        except Exception as exc:  # noqa: BLE001 - fall back, never block
            log.warning("LLM lyrics failed (%s); using the local engine", exc)
            lines = []
    # The writer's words are the words.  A line the fitter finds nothing to
    # sing in - punctuation, an empty answer - is kept as written, with no
    # notes, and the version counts it as unfitted so nobody approves it
    # unread.  It used to be replaced one for one by lexicon syllables,
    # which is how a whole Tamil lyric turned into vocables without a word
    # of warning: native script counted as zero syllables.
    sources: List[str] = [writer] * len(lines)
    for text in lines:
        if not count_syllables(text):
            log.warning("a lyric line has nothing the fitter can sing, kept as "
                        "written: %r", text[:40])
    if len(lines) < len(wanted):
        local = generate_lines(wanted, brief, seed)
        lines = lines + local[len(lines):]
        sources = sources + ["lexicon"] * (len(lines) - len(sources))

    # Lay the new lines back onto the whole song by slot, so unselected
    # phrases keep the words they had.  fit_lines takes it from here and
    # preserves locked lines by slot index as it already did.
    whole = [""] * len(slots)
    whole_sources = [""] * len(slots)
    if previous is not None:
        for i, line in enumerate(previous.lines):
            if i < len(whole):
                whole[i] = line.text
                whole_sources[i] = line.source
    for position, index in enumerate(targets):
        whole[index] = lines[position] if position < len(lines) else ""
        whole_sources[index] = sources[position] if position < len(sources) else ""
    # Every targeted slot asked for words; a blank answer there is a line
    # the singer was given nothing of, not a slot left alone on purpose.
    return fit_lines(whole, melody, brief.language, version=version,
                     previous=previous, sources=whole_sources, required=targets)


def regenerate_line(lyrics: LyricsVersion, melody: MelodyVersion, line_id: str,
                    brief: CreativeBrief, seed: int = 0, llm=None) -> List[str]:
    """Rewrite one line to the same slot; every other line is untouched."""
    from .fitting import refit_line
    slots = build_slots(melody)
    index = [l.id for l in lyrics.lines].index(line_id)
    slot = slots[index]
    text = ""
    source = ""
    if llm is not None and getattr(llm, "available", False):
        try:
            got = llm.write_lyrics([slot], brief)
            text = got[0] if got else ""
            source = f"llm:{getattr(llm, 'name', 'llm')}"
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM line rewrite failed: %s", exc)
    if not text:
        pool = _language_pool(brief.language)
        words: List[str] = []
        for t in _themes_for(brief):
            words.extend(pool.get(t, []))
        text = make_line(slot.syllable_count, words,
                         random.Random(seed or index * 7919 + 13))
        source = "lexicon"
    warnings = refit_line(lyrics, melody, line_id, text)
    line = lyrics.line_by_id(line_id)
    if line is not None:
        line.source = source
    return warnings
