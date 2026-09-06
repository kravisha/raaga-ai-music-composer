"""Words the engine does not know yet, and what becomes of them.

Specified by Arya, 2026-09-06 13:16:07 -04:00, in the shared conversation
record.  The requirement, in one line: a mood word nobody taught the engine
must not be silently discarded, and must not stop the request either.

What used to happen is the worst of both.  ``target_vector`` looks each word
up in the lexicon and skips what it does not find, so "a nervy, driving
theme" was scored as "driving" and the rest fell on the floor - no error, no
mention, a narrower brief than the creator wrote.  This module is the other
half of that lookup: the words that fell through are collected, kept, and
worked on.

Three rules shape the design.

*The request is not held up.*  Detection is a set membership test over the
same lexicons the vector already uses.  Nothing is fetched, nothing is
asked; a brief with unknown words ranks on the words that are known and
says which ones it set aside.

*Nothing is invented.*  An unknown word is resolved by mapping it onto
vocabulary the engine already has - "nervy" becomes "nervous" - and the
mapping is rejected unless every word it names is already in the lexicon.
So a resolution can never conjure an emotion weight out of a model's
imagination; the most it can do is point at a meaning that was already
there.  That is also why a resolution is useful the moment it lands: it
inherits a vector that was already trusted.

*Uncertainty is carried, not dropped.*  A mapping proposed by a model is
recorded as ``generated`` under ``core.provenance``, with the reasoning it
gave and the backend that gave it, and it stays distinguishable from one a
person confirmed.  Section 2.5 of the training specification applies here
as much as it does to phrases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core import provenance
from ..core.logging_setup import get_logger
from . import emotion, selection

log = get_logger("vocabulary")

#: Where an unresolved term is in its life.  ``NEEDS_USER_INPUT`` is the end
#: of the automatic road, not a failure: investigation ran and could not
#: settle it, so a person is the next step (specification: "ask Krish only
#: after independent investigation is insufficient").
PENDING = "pending"
INVESTIGATING = "investigating"
RESOLVED = "resolved"
NEEDS_USER_INPUT = "needs_user_input"
DISMISSED = "dismissed"

STATUSES: Tuple[str, ...] = (PENDING, INVESTIGATING, RESOLVED,
                             NEEDS_USER_INPUT, DISMISSED)

#: How many investigations may fail before a person is asked.
MAX_ATTEMPTS = 3

#: Words that carry no feeling of their own and must not be reported as
#: unknown.  The scorer already treats these specially - negators flip a
#: sense, amplifiers scale it - so they are understood, just not looked up.
_GRAMMAR = (set(emotion._NEGATORS) | set(emotion._AMPLIFIERS)
            | set(emotion._DAMPENERS))

_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "for", "with",
    "but", "or", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "as", "by", "from", "like", "more",
    "most", "some", "any", "then", "than", "so", "up", "down", "out",
    "song", "tune", "music", "feel", "feeling", "mood", "sound", "sounds",
    "something", "somethings", "kind", "sort", "bit", "make", "makes",
    "want", "wants", "need", "needs", "should", "would", "could", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "his", "her",
    "them", "they", "there", "here", "very", "really", "quite", "just",
}


def _known(word: str) -> bool:
    """Whether anything in the engine already understands this word."""
    return word in emotion.LEXICON or word in selection.FEEL_LEXICON \
        or word in selection.TEMPO_HINTS


def known_vocabulary() -> List[str]:
    """The words a resolution is allowed to map onto.

    Deliberately the emotion lexicon only.  The point of a mapping is to
    inherit a *feeling*, and the feel lexicon holds raaga tags while the
    tempo hints hold speeds; neither would give an unknown word a vector.
    """
    return sorted(emotion.LEXICON)


def unknown_terms(*texts: str, raagas=None) -> List[str]:
    """The words in a brief that nothing in the engine can read.

    Order is first-appearance, deduplicated, so the report reads the way
    the creator typed it.  A raaga named in the text is not an unknown mood
    word, which is why ``raagas`` can be passed in.
    """
    blob = " ".join(t for t in texts if t).lower()
    named = set()
    if raagas is not None:
        for name in getattr(raagas, "names", lambda: [])():
            named.update(str(name).lower().split())
    out: List[str] = []
    for token in re.findall(r"[a-z][a-z'-]{2,}", blob):
        word = token.strip("'-")
        if len(word) < 3 or word in out:
            continue
        if word in _STOPWORDS or word in _GRAMMAR or word in named:
            continue
        if _known(word):
            continue
        out.append(word)
    return out


@dataclass
class UnresolvedTerm:
    """One word the engine could not read, and the state of doing so."""

    term: str = ""
    status: str = PENDING
    #: Known words this term was found to mean.  Empty until resolved.
    mapped_to: List[str] = field(default_factory=list)
    confidence: float = 0.0
    #: Why - the model's reasoning, or a person's note.  Kept because a
    #: mapping with no stated reason cannot be reviewed later.
    evidence: str = ""
    origin: str = provenance.GENERATED
    occurrences: int = 1
    attempts: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    note: str = ""

    @property
    def usable(self) -> bool:
        """Whether a raaga search may act on this term."""
        return self.status == RESOLVED and bool(self.mapped_to)

    def describe(self) -> str:
        if self.status == RESOLVED:
            how = provenance.describe(self.origin)
            return (f"{self.term} -> {', '.join(self.mapped_to)} "
                    f"({how}, confidence {self.confidence:.2f})")
        if self.status == NEEDS_USER_INPUT:
            return f"{self.term} - {self.attempts} attempt(s), needs a person"
        return f"{self.term} - {self.status}"


def resolve_text(text: str, resolutions: Dict[str, Sequence[str]]) -> str:
    """Rewrite a brief's words into vocabulary the engine understands.

    Substitution rather than a second lexicon: once "nervy" is known to mean
    "nervous", every part of the engine that reads words - the emotion
    vector, the raaga tags, the tempo hint - gets the benefit without any of
    them learning about unresolved terms at all.
    """
    if not resolutions or not text:
        return text

    def swap(match: "re.Match[str]") -> str:
        word = match.group(0)
        known = resolutions.get(word.lower())
        return " ".join(known) if known else word

    return re.sub(r"[A-Za-z][A-Za-z'-]*", swap, text)


# --------------------------------------------------------------------------
# investigation
# --------------------------------------------------------------------------
def validate_mapping(words: Iterable[str]) -> List[str]:
    """Keep only what the engine already knows.

    This is the rule that stops a resolution from being an invention.  A
    model asked for the nearest known feeling can still answer with a word
    of its own making; that answer is not wrong so much as unusable, and it
    is dropped rather than added to the lexicon.
    """
    out: List[str] = []
    for word in words or ():
        cleaned = str(word).strip().lower()
        if cleaned in emotion.LEXICON and cleaned not in out:
            out.append(cleaned)
    return out


def investigate(term: str, llm, *, vocabulary: Optional[Sequence[str]] = None
                ) -> Tuple[List[str], float, str]:
    """Ask what an unknown mood word means, in words already understood.

    Returns ``(mapped_to, confidence, evidence)``.  An empty mapping means
    the attempt failed - no backend, a refusal, or an answer that named
    nothing the engine knows - and the caller decides whether that is worth
    another try or a question to a person.
    """
    if llm is None or not getattr(llm, "available", False):
        return [], 0.0, "no language backend was available"
    catalog = list(vocabulary or known_vocabulary())
    try:
        answer = llm.map_mood_word(term, catalog)
    except Exception as exc:  # noqa: BLE001
        log.warning("investigating %r failed: %s", term, exc)
        return [], 0.0, f"the attempt failed: {exc}"

    mapped = validate_mapping(answer.get("means", ()))
    reason = str(answer.get("reason", "")).strip()
    try:
        confidence = float(answer.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if not mapped:
        return [], 0.0, reason or "the answer named no word the engine knows"
    # A model's own confidence is not evidence about the world; it is capped
    # so that a resolution nobody has confirmed cannot outrank vocabulary
    # that was curated by hand.
    return mapped, min(confidence or 0.5, 0.8), reason
