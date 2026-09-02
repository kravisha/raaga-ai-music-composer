"""Making one thing look like one thing - specification sections 8 and 15.

The identity problem in this domain is spelling.  A raga arrives as
Shankarabharanam, Sankarabharanam or Dheerashankarabharanam; a swara as R2 or
ri; an arohanam with commas, hyphens or nothing between its notes.  Left alone
that produces a Knowledge Base with the same fact in it eleven times and no
way to see that it is the same fact.

Three jobs here:

``normalise_name``      collapse the ways one name gets transliterated, for
                        *matching only* - never for storing or displaying.
``canonical_swaras``    parse a written scale into tokens, whatever the writer
                        used to separate them.
``identity_of``         the duplicate-control key: what makes two claims the
                        same claim, which is the subject, the predicate and
                        the raga they are about - not the wording.

The last is the important one.  Two teachers describing Kambhoji's arohanam in
different words are making the same claim and should meet as a duplicate or a
contradiction; the same words about a different raga are not.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Where Roman transliterations of Indian names actually disagree.
_ASPIRATES = (("bh", "b"), ("dh", "d"), ("th", "t"), ("gh", "g"),
              ("kh", "k"), ("ph", "p"), ("jh", "j"), ("chh", "ch"),
              ("sh", "s"))

_SWARA_WORDS = {
    "sa": "S", "ri": "R2", "ru": "R2", "ga": "G3", "gi": "G3", "ma": "M1",
    "mi": "M1", "pa": "P", "dha": "D2", "da": "D2", "ni": "N3", "nu": "N3",
}
_SWARA_TOKEN = re.compile(r"^[SRGMPDN][1-3]?[+\-]*$", re.IGNORECASE)


def normalise_name(name: str) -> str:
    """Collapse transliteration differences, for matching only.

    Deliberately blunt.  It exists so that a creator typing "Kamboji" reaches
    the same entity as one typing "Kambhoji", and it is never used to rename
    anything: the canonical spelling a source used is what gets stored and
    shown.
    """
    text = (name or "").lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    for pair, replacement in _ASPIRATES:
        text = text.replace(pair, replacement)
    text = text.replace("w", "v").replace("y", "i")
    text = re.sub(r"(.)\1+", r"\1", text)               # doubled letters
    text = re.sub(r"[aeiou]+", lambda m: m.group(0)[0], text)
    return text


def normalise_predicate(predicate: str) -> str:
    """One spelling per relation-like predicate."""
    text = (predicate or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    aliases = {
        "arohana": "arohanam", "aroha": "arohanam", "ascent": "arohanam",
        "avarohana": "avarohanam", "avaroha": "avarohanam",
        "descent": "avarohanam", "jiva": "jeeva", "jeeva_swaras": "jeeva",
        "nyasa_swaras": "nyasa", "characteristic_phrase": "prayoga",
        "prayogas": "prayoga", "phrases": "prayoga", "gamakas": "gamaka",
        "mood": "rasa", "moods": "rasa", "speed": "tempo",
    }
    return aliases.get(text, text)


def canonical_swaras(value: Any) -> List[str]:
    """Parse a written scale or phrase into swara tokens.

    Accepts "S R2 G3 M1", "S,R2,G3", "s-r2-g3", ``["S", "R2"]`` and the spoken
    "sa ri ga ma".  Anything it cannot read comes back empty rather than
    half-parsed, because a half-read scale is worse than an unread one.
    """
    if isinstance(value, (list, tuple)):
        tokens = [str(t) for t in value]
    else:
        tokens = re.split(r"[\s,;|/\-]+", str(value or ""))
    out: List[str] = []
    for raw in tokens:
        token = raw.strip().strip(".")
        if not token:
            continue
        if _SWARA_TOKEN.match(token):
            head = token[0].upper()
            rest = token[1:]
            out.append(head + rest)
            continue
        spoken = _SWARA_WORDS.get(token.lower())
        if spoken:
            out.append(spoken)
            continue
        return []                # something in here is not a swara at all
    return out


def swara_base(token: str) -> str:
    """The swara without its octave marks."""
    return (token or "").replace("+", "").replace("-", "")


def normalise_statement(statement: str) -> str:
    """For comparing two wordings, not for storing one."""
    text = (statement or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .")


#: Predicates a subject may hold many values of at once.
#:
#: This distinction is load-bearing.  A raga has exactly one arohanam, so a
#: second, different one is a contradiction.  A raga has *many* characteristic
#: phrases, and the second is simply another one - treating it as a
#: contradiction of the first would fill the store with conflicts between
#: things that were never in competition, and make every real disagreement
#: harder to see.
SET_VALUED_PREDICATES = frozenset({
    "prayoga", "phrase", "example", "gamaka", "sangati", "variation",
    "composition", "avoid", "forbidden", "mistake", "alias", "exercise",
    "pattern", "tag",
})


def is_set_valued(predicate: str) -> bool:
    return normalise_predicate(predicate) in SET_VALUED_PREDICATES


def identity_of(subject: str, predicate: str, raga: str = "",
                tala: str = "", scope_hint: str = "",
                value: Any = None) -> str:
    """The duplicate-control key - section 15.

    Two items share an identity when they say something about the *same
    property of the same thing*.  Wording is deliberately not part of it: that
    is what lets one teacher's phrasing meet another's as a duplicate or a
    contradiction rather than as two unrelated rows.

    For a set-valued predicate the value *is* part of the identity, because
    each value is a separate thing to know rather than a rival answer to one
    question.  Two sources offering the same phrase still meet as duplicates;
    two different phrases simply coexist.
    """
    predicate_key = normalise_predicate(predicate)
    parts = [normalise_name(subject), predicate_key, normalise_name(raga),
             normalise_name(tala), normalise_name(scope_hint)]
    if predicate_key in SET_VALUED_PREDICATES and value is not None:
        swaras = canonical_swaras(value)
        parts.append(" ".join(swaras) if swaras
                     else normalise_statement(str(value)))
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:24]


def source_identity(reference: str, title: str = "") -> str:
    """One identity per source, through any of the links that reach it."""
    text = (reference or "").strip().lower()
    if text:
        text = re.sub(r"^https?://", "", text)
        text = re.sub(r"^(www|m)\.", "", text)
        short = re.match(r"youtu\.be/([\w-]+)", text)
        if short:
            return f"youtube:{short.group(1)}"
        watch = re.match(r"youtube\.com/watch\?.*\bv=([\w-]+)", text)
        if watch:
            return f"youtube:{watch.group(1)}"
        text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        return f"ref:{text}"
    return "title:" + hashlib.sha1(
        (title or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def structured_for(predicate: str, value: Any) -> Dict[str, Any]:
    """The machine-readable half of a claim - section 4.

    A statement a person can read is not enough on its own: composition needs
    the value in a form it can act on.  Where the predicate has a known shape,
    it is parsed into it; where it does not, the raw value is kept as given
    rather than forced into a shape it does not have.
    """
    predicate = normalise_predicate(predicate)
    if predicate in ("arohanam", "avarohanam", "jeeva", "nyasa", "graha",
                     "prayoga", "swaras"):
        swaras = canonical_swaras(value)
        if swaras:
            return {"kind": "swaras", "swaras": swaras, "count": len(swaras)}
        return {"kind": "text", "value": str(value)}
    if predicate in ("tempo", "beat_count", "confidence"):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
        if numbers:
            return {"kind": "number", "value": float(numbers[0]),
                    "range": [float(n) for n in numbers[:2]]}
        return {"kind": "text", "value": str(value)}
    if predicate in ("rasa", "tags", "aliases"):
        items = [p.strip() for p in re.split(r"[,;]", str(value)) if p.strip()]
        return {"kind": "list", "items": items}
    if isinstance(value, dict):
        return {"kind": "object", **value}
    return {"kind": "text", "value": str(value)}


def similarity(a: str, b: str) -> float:
    """How alike two statements are, as a share of the words they share.

    Used only to *offer* a near duplicate for merging, never to merge one
    silently: section 15 wants near duplicates linked or merged deliberately.
    """
    first = set(normalise_statement(a).split())
    second = set(normalise_statement(b).split())
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def alias_variants(name: str) -> List[str]:
    """Spellings worth indexing for a name, including the one given."""
    name = (name or "").strip()
    if not name:
        return []
    variants = {name}
    # The commonest single difference in practice: an aspirate written or not.
    for pair, replacement in _ASPIRATES:
        if pair in name.lower():
            variants.add(re.sub(pair, replacement, name, flags=re.IGNORECASE))
    return sorted(variants)
