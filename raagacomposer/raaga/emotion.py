"""Brief to raaga, through an emotion vector - Stage 1 pack document 05.

The pack's argument, and the reason this file exists: a raaga should be
suggested because its *blocks* fit what was asked for, not because its name
is filed under "sad".  Every melakarta decomposes into an R-G block, a
madhyama and a D-N block (pack document 01 sections C to F), each of which
carries a starter character; a brief decomposes into fourteen emotion
dimensions; and the match between the two is something a person can be shown
rather than asked to trust.

Three things live here:

``target_vector``   what the creative brief is asking for, as fourteen
                    weights, blended over the brief's fields by the pack's
                    own weights (document 05 section 3).
``profile_vector``  what a raaga offers, built from its block characters,
                    its starter tags and any curated moods.
``rank``            the two compared, with the pack's contradiction
                    penalties and block bonuses, then spread for diversity.

Everything is deterministic and offline.  The pack allows a semantic
classifier to build the target vector where one is configured; that is a
provider task and this module is the fallback it falls back to, so the same
scoring runs either way and the application still ranks properly with no key
and nothing installed.

Hard knowledge is not in this file.  Scales and blocks are grammar and live
in the library; everything here is the pack's ``[HEURISTIC]`` layer, which
is why it can be overridden by learned weights later (S3) without any of it
touching a raaga's notes.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.models import CreativeBrief
from .library import Raaga

#: Pack document 05 section 1.  Order is fixed so a vector can be printed,
#: compared and stored as a row without a key each time.
DIMENSIONS: Tuple[str, ...] = (
    "sadness", "tenderness", "yearning", "romance", "devotion", "serenity",
    "joy", "warmth", "brightness", "gravity", "mystery", "tension", "power",
    "wonder",
)

#: How much each brief field says about the target (document 05 section 3).
#: Weights are renormalised over the fields that are actually filled in, so a
#: brief with only a mood is still a brief and not a mostly-empty vector.
FIELD_WEIGHTS: Dict[str, float] = {
    "mood": 0.30, "feel": 0.30, "situation": 0.25, "title": 0.10,
    "notes": 0.05,
}

_NEGATORS = {"not", "no", "never", "without", "isnt", "arent", "wasnt",
             "dont", "doesnt", "hardly", "barely", "less", "nothing"}
_AMPLIFIERS = {"very": 1.2, "deeply": 1.2, "utterly": 1.25, "so": 1.1,
               "really": 1.1, "extremely": 1.25, "too": 1.15, "intensely": 1.2}
_DAMPENERS = {"slightly": 0.6, "little": 0.6, "bit": 0.6, "somewhat": 0.7,
              "faintly": 0.55, "mildly": 0.6, "gently": 0.8}


def _v(**weights: float) -> Dict[str, float]:
    for name in weights:
        if name not in DIMENSIONS:
            raise KeyError(f"{name!r} is not one of the fourteen dimensions")
    return dict(weights)


#: One vocabulary for both sides of the comparison.  A creator's "lonely" and
#: the pack's "plaintive" have to land in the same space or there is nothing
#: to compare, so this table covers three vocabularies at once: the words
#: creators type, the block characters in pack documents 01 and 02 to 04, and
#: the "good for" uses those files list.
LEXICON: Dict[str, Dict[str, float]] = {
    # -- what a creator types ---------------------------------------------
    "sad": _v(sadness=1.0, gravity=0.35),
    "sadness": _v(sadness=1.0, gravity=0.35),
    "sorrow": _v(sadness=1.0, gravity=0.5),
    "grief": _v(sadness=1.0, gravity=0.7, yearning=0.5),
    "grieving": _v(sadness=1.0, gravity=0.7, yearning=0.5),
    "cry": _v(sadness=0.9, tenderness=0.3),
    "tears": _v(sadness=0.9, tenderness=0.3),
    "melancholy": _v(sadness=0.85, tenderness=0.4, serenity=0.25),
    "wistful": _v(sadness=0.6, tenderness=0.55, yearning=0.5),
    "bittersweet": _v(sadness=0.6, tenderness=0.6, warmth=0.45, yearning=0.5),
    "nostalgia": _v(sadness=0.5, tenderness=0.6, warmth=0.5, yearning=0.6),
    "nostalgic": _v(sadness=0.5, tenderness=0.6, warmth=0.5, yearning=0.6),
    "lonely": _v(sadness=0.75, yearning=0.8, gravity=0.35),
    "loneliness": _v(sadness=0.75, yearning=0.8, gravity=0.35),
    "alone": _v(sadness=0.55, yearning=0.6, gravity=0.3),
    "longing": _v(yearning=1.0, tenderness=0.5, sadness=0.4),
    "yearning": _v(yearning=1.0, tenderness=0.5, sadness=0.4),
    "missing": _v(yearning=0.9, sadness=0.5, tenderness=0.45),
    "separation": _v(yearning=0.9, sadness=0.7, gravity=0.4),
    "ache": _v(yearning=0.85, sadness=0.6, tenderness=0.4),
    "heartbreak": _v(sadness=0.9, yearning=0.85, tenderness=0.4),
    "romantic": _v(romance=1.0, tenderness=0.6, warmth=0.6),
    "romance": _v(romance=1.0, tenderness=0.6, warmth=0.6),
    "love": _v(romance=0.9, tenderness=0.6, warmth=0.6),
    "intimate": _v(tenderness=0.85, warmth=0.6, romance=0.5, serenity=0.3),
    "tender": _v(tenderness=1.0, warmth=0.5),
    "gentle": _v(tenderness=0.8, serenity=0.6, warmth=0.4),
    "soft": _v(tenderness=0.75, serenity=0.55, warmth=0.4),
    "warm": _v(warmth=1.0, tenderness=0.5),
    "warmth": _v(warmth=1.0, tenderness=0.5),
    "devotional": _v(devotion=1.0, gravity=0.5, serenity=0.4),
    "devotion": _v(devotion=1.0, gravity=0.5, serenity=0.4),
    "prayer": _v(devotion=0.95, serenity=0.5, gravity=0.4),
    "prayerful": _v(devotion=0.95, serenity=0.5, gravity=0.4),
    "temple": _v(devotion=0.85, gravity=0.5, serenity=0.35),
    "sacred": _v(devotion=0.85, gravity=0.55, wonder=0.35),
    "spiritual": _v(devotion=0.85, wonder=0.45, mystery=0.35),
    "calm": _v(serenity=1.0, tenderness=0.3),
    "peaceful": _v(serenity=1.0, tenderness=0.3),
    "serene": _v(serenity=1.0),
    "still": _v(serenity=0.8, mystery=0.3),
    "meditative": _v(serenity=0.8, devotion=0.5, mystery=0.4),
    "contemplative": _v(serenity=0.7, gravity=0.45, devotion=0.3),
    "reflective": _v(serenity=0.6, gravity=0.45, tenderness=0.35),
    "happy": _v(joy=1.0, brightness=0.7, warmth=0.5),
    "joy": _v(joy=1.0, brightness=0.7, warmth=0.5),
    "joyful": _v(joy=1.0, brightness=0.7, warmth=0.5),
    "playful": _v(joy=0.8, brightness=0.6, warmth=0.4),
    "celebration": _v(joy=0.95, brightness=0.8, power=0.4),
    "celebratory": _v(joy=0.95, brightness=0.8, power=0.4),
    "festive": _v(joy=0.9, brightness=0.8, power=0.4),
    "wedding": _v(joy=0.85, brightness=0.7, warmth=0.6, devotion=0.35),
    "auspicious": _v(brightness=0.8, joy=0.6, devotion=0.5),
    "bright": _v(brightness=1.0, joy=0.5),
    "brightness": _v(brightness=1.0, joy=0.5),
    "radiant": _v(brightness=1.0, joy=0.55, wonder=0.4),
    "hopeful": _v(brightness=0.7, joy=0.5, warmth=0.5, yearning=0.35),
    "hope": _v(brightness=0.7, joy=0.5, warmth=0.5, yearning=0.35),
    "energetic": _v(power=0.8, brightness=0.7, joy=0.6),
    "dance": _v(joy=0.8, brightness=0.7, power=0.5),
    "grand": _v(power=0.9, brightness=0.6, gravity=0.5),
    "majestic": _v(power=0.9, gravity=0.6, brightness=0.55),
    "heroic": _v(power=1.0, brightness=0.5, tension=0.4),
    "brave": _v(power=0.85, brightness=0.45),
    "battle": _v(power=0.9, tension=0.8, gravity=0.5),
    "triumphant": _v(power=0.9, joy=0.6, brightness=0.7),
    "noble": _v(gravity=0.7, power=0.6, devotion=0.35),
    "dignified": _v(gravity=0.75, power=0.5, serenity=0.3),
    "solemn": _v(gravity=0.9, devotion=0.5),
    "grave": _v(gravity=1.0, sadness=0.4),
    "serious": _v(gravity=0.8),
    "dark": _v(gravity=0.7, mystery=0.5, sadness=0.5),
    "brooding": _v(gravity=0.7, tension=0.6, sadness=0.5),
    "mystical": _v(mystery=1.0, wonder=0.6, devotion=0.35),
    "mystery": _v(mystery=1.0, wonder=0.55),
    "mysterious": _v(mystery=1.0, wonder=0.55),
    "eerie": _v(mystery=0.9, tension=0.6),
    "dream": _v(mystery=0.6, wonder=0.6, tenderness=0.4),
    "dreamy": _v(mystery=0.6, wonder=0.6, tenderness=0.4),
    "wonder": _v(wonder=1.0, brightness=0.4),
    "awe": _v(wonder=0.95, gravity=0.5, power=0.4),
    "magical": _v(wonder=0.9, mystery=0.7, brightness=0.4),
    "tense": _v(tension=1.0),
    "tension": _v(tension=1.0),
    "anxious": _v(tension=0.9, sadness=0.35),
    "restless": _v(tension=0.8, power=0.45),
    "urgent": _v(tension=0.85, power=0.6),
    "angry": _v(tension=0.85, power=0.7, gravity=0.4),
    "intense": _v(tension=0.7, power=0.8, gravity=0.4),
    "dramatic": _v(tension=0.65, power=0.7, gravity=0.5),
    "night": _v(mystery=0.45, serenity=0.35, gravity=0.3),
    "midnight": _v(mystery=0.55, serenity=0.3, gravity=0.35),
    "dawn": _v(brightness=0.6, serenity=0.6, devotion=0.35),
    "morning": _v(brightness=0.6, serenity=0.55, joy=0.35),
    "evening": _v(warmth=0.5, serenity=0.45, tenderness=0.35),
    "rain": _v(tenderness=0.5, serenity=0.4, yearning=0.4),
    "village": _v(warmth=0.6, serenity=0.45, joy=0.35),
    "childhood": _v(warmth=0.6, joy=0.5, tenderness=0.5, yearning=0.4),
    "lullaby": _v(tenderness=0.9, serenity=0.7, warmth=0.6),
    "farewell": _v(sadness=0.7, yearning=0.7, gravity=0.45),
    "loss": _v(sadness=0.85, gravity=0.6, yearning=0.5),
    "failure": _v(sadness=0.7, gravity=0.5),
    "betrayal": _v(sadness=0.7, tension=0.6, gravity=0.5),

    # -- the pack's block characters (documents 01 C to E, 02 to 04) -------
    "compressed": _v(tension=0.8, gravity=0.5, mystery=0.3),
    "austere": _v(gravity=0.85, devotion=0.4, tension=0.35),
    "vivadi": _v(tension=0.7, wonder=0.4),
    "colored": {},
    "plaintive": _v(sadness=0.8, yearning=0.75, tenderness=0.5),
    "inward": _v(yearning=0.6, gravity=0.5, serenity=0.35),
    "contrast": _v(tension=0.4, wonder=0.35),
    "wide": _v(power=0.45, wonder=0.35),
    "introspective": _v(gravity=0.5, tenderness=0.5, serenity=0.4),
    "humane": _v(tenderness=0.8, warmth=0.6),
    "compassionate": _v(tenderness=0.85, warmth=0.6, devotion=0.4),
    "open": _v(brightness=0.6, joy=0.45, warmth=0.5),
    "lyrical": _v(tenderness=0.6, romance=0.55, warmth=0.5),
    "confident": _v(power=0.6, brightness=0.55, joy=0.35),
    "edged": _v(tension=0.6, brightness=0.5),
    "grounded": _v(serenity=0.6, gravity=0.5),
    "earthy": _v(warmth=0.5, serenity=0.5, gravity=0.35),
    "settled": _v(serenity=0.75, gravity=0.35),
    "luminous": _v(brightness=0.8, wonder=0.6, mystery=0.4),
    "searching": _v(yearning=0.7, mystery=0.6, wonder=0.45),
    "heightened": _v(tension=0.6, power=0.5, wonder=0.4),
    "unresolved": _v(tension=0.8, yearning=0.55, mystery=0.4),
    "pathos": _v(sadness=0.9, tenderness=0.55, gravity=0.45),
    "descending": _v(sadness=0.4, gravity=0.35),
    "poignant": _v(sadness=0.7, yearning=0.75, tenderness=0.55),
    "upward": _v(brightness=0.5, power=0.4, yearning=0.35),
    "pull": {},
    "resolution": _v(brightness=0.5, serenity=0.45, power=0.35),
    "rounded": _v(warmth=0.7, serenity=0.5, tenderness=0.45),
    "relaxed": _v(serenity=0.8, warmth=0.5),
    "affirmative": _v(brightness=0.7, joy=0.6, power=0.5),
    "expansive": _v(power=0.7, brightness=0.6, wonder=0.5),
    "unusual": _v(wonder=0.6, mystery=0.5, tension=0.35),
    "experimental": _v(wonder=0.7, mystery=0.5, tension=0.35),
    "severe": _v(gravity=0.8, tension=0.6),
    "unease": _v(tension=0.8, mystery=0.45),
    "ritual": _v(devotion=0.85, gravity=0.6),
    "solemnity": _v(gravity=0.9, devotion=0.55),
    "majesty": _v(power=0.85, gravity=0.6, brightness=0.5),
    "disciplined": _v(gravity=0.7, serenity=0.4, devotion=0.4),
    "virtuosic": _v(power=0.8, tension=0.5, wonder=0.5),
    "restlessness": _v(tension=0.8, power=0.45),
    "nobility": _v(gravity=0.7, power=0.6),
    "elegance": _v(tenderness=0.5, brightness=0.5, serenity=0.4),
    "anguish": _v(sadness=0.95, tension=0.7, yearning=0.7),
    "tragic": _v(sadness=0.95, gravity=0.75),
    "pleading": _v(yearning=0.85, devotion=0.5, sadness=0.5),
    "grandeur": _v(power=0.85, gravity=0.6, wonder=0.45),
    "profound": _v(gravity=0.85, devotion=0.45),
    "rich": _v(warmth=0.6, gravity=0.4),
    "openness": _v(brightness=0.6, joy=0.45, warmth=0.5),
    "radiance": _v(brightness=1.0, joy=0.55, wonder=0.4),
    "expansiveness": _v(power=0.7, brightness=0.6, wonder=0.5),
    "auspiciousness": _v(brightness=0.8, joy=0.6, devotion=0.5),
    "brilliant": _v(brightness=0.9, power=0.6, wonder=0.45),
    "urgency": _v(tension=0.85, power=0.6),
}

#: Multi-word keys, checked against the raw text before tokens are matched.
PHRASES: Dict[str, Dict[str, float]] = {
    "love failure": _v(sadness=0.85, yearning=0.85, romance=0.6, gravity=0.45),
    "late night": _v(mystery=0.5, serenity=0.35, gravity=0.3),
    "late at night": _v(mystery=0.5, serenity=0.35, gravity=0.3),
    "broken heart": _v(sadness=0.9, yearning=0.85, romance=0.5),
    "first love": _v(romance=0.9, tenderness=0.7, warmth=0.6, joy=0.4),
    "coming home": _v(warmth=0.8, joy=0.5, tenderness=0.5),
    "letting go": _v(sadness=0.7, yearning=0.6, serenity=0.4),
    "growing up": _v(warmth=0.5, yearning=0.5, tenderness=0.45),
}

_WORD = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------
# the vector
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EmotionVector:
    """Fourteen weights in 0..1.  A shape, not a measurement."""

    weights: Dict[str, float] = field(default_factory=dict)

    def __getitem__(self, dimension: str) -> float:
        return self.weights.get(dimension, 0.0)

    @property
    def empty(self) -> bool:
        return not any(v > 0.0 for v in self.weights.values())

    def scaled(self, factor: float) -> "EmotionVector":
        return EmotionVector({k: v * factor for k, v in self.weights.items()})

    def normalised(self) -> "EmotionVector":
        """Rescaled so the strongest dimension is 1.0, shape preserved."""
        top = max(self.weights.values(), default=0.0)
        if top <= 0:
            return EmotionVector({})
        return EmotionVector({k: round(min(1.0, v / top), 3)
                              for k, v in self.weights.items() if v > 0})

    def dominant(self, count: int = 3, floor: float = 0.25) -> List[str]:
        ranked = sorted(((v, k) for k, v in self.weights.items() if v >= floor),
                        key=lambda pair: (-pair[0], pair[1]))
        return [k for _, k in ranked[:count]]

    def similarity(self, other: "EmotionVector") -> float:
        """Cosine similarity: how alike the two shapes are, 0..1.

        Cosine rather than overlap because both sides are shapes with
        arbitrary overall loudness - a brief that says "sad" once and one
        that says "very sad indeed" are asking for the same thing.
        """
        keys = set(self.weights) | set(other.weights)
        dot = sum(self[k] * other[k] for k in keys)
        left = math.sqrt(sum(self[k] ** 2 for k in keys))
        right = math.sqrt(sum(other[k] ** 2 for k in keys))
        if left <= 0 or right <= 0:
            return 0.0
        return dot / (left * right)

    def describe(self, count: int = 3) -> str:
        return ", ".join(self.dominant(count)) or "nothing in particular"


def _combine(into: Dict[str, float], addition: Dict[str, float],
             factor: float = 1.0) -> None:
    """Strongest claim wins per dimension, rather than a sum.

    Two words that both mean "sad" describe one sadness; summing them would
    make a repetitive brief look more certain than a precise one.
    """
    for dimension, weight in addition.items():
        value = min(1.0, weight * factor)
        if value > into.get(dimension, 0.0):
            into[dimension] = value


def read_text(text: str) -> EmotionVector:
    """What one piece of free text is asking for.

    Negation is honoured by suppression, not inversion: "not sad" is a
    statement that sadness is not wanted, and guessing which of the other
    thirteen dimensions the creator did mean would be inventing a brief they
    did not write.
    """
    low = (text or "").lower()
    if not low.strip():
        return EmotionVector({})

    found: Dict[str, float] = {}
    for phrase, vector in PHRASES.items():
        if phrase in low:
            _combine(found, vector)
            low = low.replace(phrase, " ")

    tokens = _WORD.findall(low)
    for index, token in enumerate(tokens):
        vector = LEXICON.get(token)
        if vector is None:
            # A creator writes "lonely" and also "loneliness"; match a longer
            # typed word back to a known stem, never the other way round, so
            # a short word cannot claim a long entry.
            for key, candidate in LEXICON.items():
                if len(key) >= 4 and token.startswith(key):
                    vector = candidate
                    break
        if not vector:
            continue
        window = tokens[max(0, index - 3):index]
        if any(w in _NEGATORS for w in window):
            continue
        factor = 1.0
        for modifier in window[-2:]:
            factor *= _AMPLIFIERS.get(modifier, _DAMPENERS.get(modifier, 1.0))
        _combine(found, vector, factor)
    return EmotionVector(found)


def target_vector(brief: CreativeBrief) -> EmotionVector:
    """What the brief is asking for, blended by the pack's field weights.

    Weights are renormalised over the fields that actually say something, so
    a brief with nothing but a mood is read as a mood rather than as a mostly
    empty vector.
    """
    fields = {
        "mood": brief.mood, "feel": brief.feel, "situation": brief.situation,
        "title": brief.title, "notes": brief.notes,
    }
    read = {name: read_text(text) for name, text in fields.items()
            if (text or "").strip()}
    read = {name: vector for name, vector in read.items() if not vector.empty}
    if not read:
        return EmotionVector({})
    total = sum(FIELD_WEIGHTS[name] for name in read) or 1.0
    blended: Dict[str, float] = {}
    for name, vector in read.items():
        share = FIELD_WEIGHTS[name] / total
        for dimension, weight in vector.weights.items():
            blended[dimension] = blended.get(dimension, 0.0) + weight * share
    return EmotionVector(blended).normalised()


def profile_vector(raaga: Raaga) -> EmotionVector:
    """What a raaga offers, from its blocks first and its curation second.

    The block characters are the pack's model and every melakarta has them.
    Starter tags and curated moods are added where they exist; a raaga that
    somebody wrote moods for is described by both, and one that arrived as a
    bare scale is described by its blocks alone - which is the whole point of
    the block model.
    """
    found: Dict[str, float] = {}
    for character in raaga.block_character.values():
        _combine(found, read_text(character).weights)
    if raaga.tags:
        _combine(found, read_text(" ".join(raaga.tags)).weights, 0.9)
    if raaga.good_for:
        _combine(found, read_text(" ".join(raaga.good_for)).weights, 0.8)
    if raaga.moods:
        # Curated by a person about this raaga in particular, so it speaks a
        # little louder than a character inherited from a block.
        _combine(found, read_text(" ".join(raaga.moods)).weights, 1.0)
    return EmotionVector(found).normalised()


# --------------------------------------------------------------------------
# scoring: pack document 05 section 3
# --------------------------------------------------------------------------
#: The blocks each rule talks about.  Named so a penalty can say which block
#: it objected to rather than quoting a number at the creator.
_URGENT_BLOCKS = {"R3G3", "D3N3"}
_DARK_UNRESOLVED = {"R1G1", "D1N1"}
_GRAVE_BLOCKS = {"R1G2", "D1N2"}
_WARM_RG = {"R2G2", "R2G3"}
_GRIEF_RG = {"R1G2", "R2G2"}
_GRIEF_DN = {"D1N2", "D1N3"}
_POIGNANT = {"R1G2", "D1N2", "D1N3"}

#: Bonuses tune the ranking; they never decide it.  Capped so that a raaga
#: the brief does not actually resemble cannot climb over one it does on
#: block rules alone - the cosine fit is the argument, these are the pack's
#: notes in the margin.
MAX_BONUS = 0.15
MAX_PENALTY = 0.25


def sentence_case(text: str) -> str:
    """Capitalise the first letter and leave the rest alone.

    ``str.capitalize`` lower-cases everything after the first character,
    which turns "the same colour over D2N3" into "...over d2n3" - the block
    names are the one thing in these sentences that must not be reworded.
    """
    return text[:1].upper() + text[1:] if text else text


@dataclass
class Scored:
    """One ranked raaga, with the working shown (pack document 05 section 4)."""

    raaga: Raaga
    fit: float                                   # cosine, 0..1
    score: float                                 # 0..100
    reason: str = ""
    tags: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)
    role: str = ""                               # closest fit, warmer, ...
    confidence: float = 0.5

    @property
    def name(self) -> str:
        return self.raaga.name


def _blocks(raaga: Raaga) -> Tuple[str, str, str]:
    return raaga.rg, raaga.madhyama, raaga.dn


def _adjustments(target: EmotionVector, raaga: Raaga
                 ) -> Tuple[float, List[str], List[str]]:
    """The pack's contradiction penalties and block bonuses.

    A raaga with no blocks - a janya nobody derived from a parent scale here
    - collects neither, and is ranked on its curated character alone.  That
    is the honest outcome: these rules are stated about blocks, and inventing
    a block for a raaga that has none would be fabricating grammar.
    """
    rg, madhyama, dn = _blocks(raaga)
    if not (rg and dn):
        return 0.0, [], []

    sad = max(target["sadness"], target["yearning"])
    tender = target["tenderness"]
    peaceful = target["serenity"]
    festive = max(target["joy"], target["brightness"])
    wants_intensity = max(target["tension"], target["power"])
    bonuses: List[str] = []
    penalties: List[str] = []
    bonus = penalty = 0.0

    # -- contradictions ----------------------------------------------------
    if max(sad, tender) > 0.5 and (rg in _URGENT_BLOCKS or dn in _URGENT_BLOCKS) \
            and wants_intensity < 0.5:
        penalty += 0.18 * max(sad, tender)
        penalties.append(f"{rg if rg in _URGENT_BLOCKS else dn} is urgent and "
                         f"bright-edged where the brief asks for something "
                         f"quieter")
    if peaceful > 0.5 and (rg in _DARK_UNRESOLVED or dn in _DARK_UNRESOLVED):
        penalty += 0.16 * peaceful
        penalties.append(f"{rg if rg in _DARK_UNRESOLVED else dn} leaves things "
                         f"tense and unresolved where the brief asks for calm")
    if festive > 0.6 and (rg in _GRAVE_BLOCKS or dn in _GRAVE_BLOCKS):
        penalty += 0.15 * festive
        penalties.append(f"{rg if rg in _GRAVE_BLOCKS else dn} is grave and "
                         f"plaintive where the brief is celebratory")

    # -- affinities --------------------------------------------------------
    if max(target["mystery"], target["wonder"]) > 0.45 and madhyama == "M2":
        bonus += 0.10
        bonuses.append("M2 is the searching, luminous madhyama")
    if target["romance"] > 0.45 and target["warmth"] > 0.4:
        if rg in _WARM_RG:
            bonus += 0.07
            bonuses.append(f"{rg} is the warm, lyrical R-G block")
        if dn == "D2N2":
            bonus += 0.06
            bonuses.append("D2N2 is warm, gentle and rounded")
    if sad > 0.5:
        if rg in _GRIEF_RG:
            bonus += 0.07
            bonuses.append(f"{rg} carries the tender, plaintive colour")
        if dn in _GRIEF_DN:
            bonus += 0.07
            bonuses.append(f"{dn} is where the pathos and the poignancy sit")
    if festive > 0.55 and target["power"] > 0.35:
        if rg == "R2G3":
            bonus += 0.06
            bonuses.append("R2G3 is open and confident")
        if dn == "D2N3":
            bonus += 0.06
            bonuses.append("D2N3 is affirmative and expansive")
    if target["devotion"] > 0.5 and wants_intensity > 0.4 and madhyama == "M2" \
            and (rg in _POIGNANT or dn in _POIGNANT):
        bonus += 0.08
        bonuses.append("M2 against a plaintive block is the intense "
                       "devotional colour")

    return (min(bonus, MAX_BONUS) - min(penalty, MAX_PENALTY),
            bonuses, penalties)


def _tempo_fit(brief: CreativeBrief, raaga: Raaga) -> float:
    """The pack's 0.05 tie-break.  Silent when nothing is known either way."""
    if not brief.tempo_preference or not raaga.tempo_range:
        return 0.0
    low, high = raaga.tempo_range[0], raaga.tempo_range[-1]
    return 0.05 if low <= brief.tempo_preference <= high else -0.03


def _fit_tags(target: EmotionVector, profile: EmotionVector) -> List[str]:
    """Two to four words for why this one, from what both sides agree on."""
    shared = [d for d in DIMENSIONS
              if target[d] >= 0.3 and profile[d] >= 0.3]
    shared.sort(key=lambda d: -(target[d] * profile[d]))
    if len(shared) < 2:
        shared += [d for d in profile.dominant(3) if d not in shared]
    return shared[:4]


def _reason(raaga: Raaga, tags: Sequence[str], bonuses: Sequence[str],
            penalties: Sequence[str]) -> str:
    """One sentence, traceable to the block map (pack document 01 section F)."""
    summary = raaga.block_summary()
    head = summary or (", ".join(raaga.moods[:3]) if raaga.moods else "")
    if not head:
        head = f"{raaga.name} as the library has it"
    wanted = ", ".join(tags[:3])
    sentence = f"{head}"
    if wanted:
        sentence += f" - which is where the {wanted} you asked for lives"
    if bonuses:
        sentence += f"; {bonuses[0]}"
    if penalties:
        sentence += f". Against it: {penalties[0]}"
    return sentence + "." if not sentence.endswith(".") else sentence


def score_raaga(brief: CreativeBrief, raaga: Raaga,
                target: Optional[EmotionVector] = None) -> Scored:
    """One raaga against one brief, with its reasons."""
    target = target if target is not None else target_vector(brief)
    profile = profile_vector(raaga)
    fit = target.similarity(profile)
    adjustment, bonuses, penalties = _adjustments(target, raaga)
    total = fit + adjustment + _tempo_fit(brief, raaga)
    tags = _fit_tags(target, profile)
    # Scaled by the best a raaga could possibly do rather than clamped at
    # 1.0.  Clamping put every good answer on 100 and threw away exactly the
    # differences the ranking exists to express - and with the scores flat,
    # the diversity step below had nothing to trade against.
    ceiling = 1.0 + MAX_BONUS + 0.05
    return Scored(
        raaga=raaga, fit=round(fit, 4),
        score=round(100.0 * max(0.0, total) / ceiling, 1),
        reason=_reason(raaga, tags, bonuses, penalties),
        tags=tags, bonuses=bonuses, penalties=penalties)


# --------------------------------------------------------------------------
# diversity: pack document 05 section 5
# --------------------------------------------------------------------------
#: How hard a near-duplicate is pushed down.  Low enough that a clearly
#: better raaga still wins its place; high enough that five raagas built from
#: the same three blocks cannot fill the list.
DIVERSITY = 0.25


def _roles(top: EmotionVector, other: EmotionVector,
           top_raaga: Raaga, raaga: Raaga) -> List[str]:
    """How this alternative differs from the first choice, best reason first.

    A list rather than one answer so that the caller can give each entry a
    description of its own when two of them differ in the same direction.
    """
    def delta(*dimensions: str) -> float:
        return sum(other[d] - top[d] for d in dimensions)

    ranked = sorted(
        ((delta("warmth", "tenderness", "serenity"), "a warmer alternative"),
         (delta("gravity", "sadness", "tension"),
          "a darker, more serious alternative"),
         (delta("brightness", "joy", "power"), "a brighter alternative"),
         (delta("mystery", "wonder"), "a stranger, more searching alternative")),
        key=lambda pair: -pair[0])
    out = [label for value, label in ranked if value > 0.1]
    if top_raaga.madhyama and raaga.madhyama and \
            top_raaga.madhyama != raaga.madhyama:
        out.insert(min(1, len(out)), f"the {raaga.madhyama} contrast")
    if raaga.dn and top_raaga.dn and raaga.dn != top_raaga.dn:
        out.append(f"the same colour over {raaga.dn}")
    out.append("a close relative")
    return out


def spread(scored: Sequence[Scored], limit: int = 5,
           diversity: float = DIVERSITY) -> List[Scored]:
    """Rank, then make sure the list is worth reading.

    Straight score order answers "which is best" five times over; when the
    top scores are close that is five near-identical profiles and no help at
    all.  Each place after the first goes to the raaga with the best score
    once its likeness to what is already chosen is discounted, so the list
    becomes the pack's spread - the closest fit, then a warmer, darker or
    brighter alternative, or the other madhyama - without any of those roles
    being hard-coded as a slot to fill.
    """
    remaining = sorted(scored, key=lambda s: (-s.score, s.name))
    if not remaining:
        return []
    chosen = [remaining.pop(0)]
    vectors = {chosen[0].name: profile_vector(chosen[0].raaga)}
    while remaining and len(chosen) < limit:
        best_index, best_value = 0, None
        for index, candidate in enumerate(remaining):
            vector = vectors.setdefault(candidate.name,
                                        profile_vector(candidate.raaga))
            likeness = max((vector.similarity(vectors[c.name]) for c in chosen),
                           default=0.0)
            value = candidate.score - 100.0 * diversity * likeness
            if best_value is None or value > best_value:
                best_index, best_value = index, value
        chosen.append(remaining.pop(best_index))
    top_vector = vectors[chosen[0].name]
    chosen[0].role = "the closest fit"
    taken = {chosen[0].role}
    for item in chosen[1:]:
        # Each alternative earns its own description; two entries both
        # labelled "a warmer alternative" tell the creator nothing about why
        # both are in the list.
        for candidate in _roles(top_vector, vectors[item.name],
                                chosen[0].raaga, item.raaga):
            if candidate not in taken:
                item.role = candidate
                break
        else:
            item.role = "another close relative"
        taken.add(item.role)
    # Diversity decides *which* five (pack section 5: "avoid five
    # almost-identical profiles"); it does not get to decide their order.  A
    # list numbered 1 to 5 whose scores do not descend reads as a defect, and
    # the agent's own acceptance test requires descending scores.  The first
    # pick is the highest scorer, so it stays first either way.
    return [chosen[0]] + sorted(chosen[1:], key=lambda s: (-s.score, s.name))


def rank(brief: CreativeBrief, raagas: Iterable[Raaga], limit: int = 5,
         diversity: float = DIVERSITY) -> List[Scored]:
    """The pack's engine end to end: brief in, a ranked spread out.

    Never returns an empty list for a brief with anything in it (pack
    document 05 section 4); a brief that says nothing at all returns nothing
    and the caller decides what to do about it, which is the one case where
    an empty answer is the honest one.
    """
    target = target_vector(brief)
    if target.empty:
        return []
    scored = [score_raaga(brief, raaga, target) for raaga in raagas]
    scored = [s for s in scored if s.fit > 0.0]
    if not scored:
        return []
    chosen = spread(scored, limit=limit, diversity=diversity)
    top = chosen[0].score if chosen else 0.0
    for item in chosen:
        # Confident when the fit is strong and the field is not a coin toss.
        item.confidence = round(min(0.95, 0.35 + 0.5 * item.fit
                                    + (0.1 if item.score >= top - 5 else 0.0)), 3)
    return chosen
