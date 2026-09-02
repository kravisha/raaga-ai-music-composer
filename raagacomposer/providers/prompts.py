"""How each task is put to a model, and how the answer is read back.

Every text backend asks the same question in the same words.  That is the
point: a local 3B model and Claude are then answering the same thing, so the
router can move a task between them without silently changing what was asked,
and a disappointing answer is the model's doing rather than the prompt's.

Parsing lives here too, because small local models are markedly less obedient
about returning bare JSON than a frontier model is, and every backend needs the
same tolerance.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

Prompt = Tuple[str, str]                     # (system, user)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------
def lyrics(slots: Sequence[Any], brief: Any) -> Prompt:
    lines = []
    for i, slot in enumerate(slots, start=1):
        pattern = "".join("X" if s else "." for s in slot.stresses)
        lines.append(f"{i}. section={slot.section_name} "
                     f"syllables={slot.syllable_count} stress={pattern}")
    system = (
        "You write song lyrics that fit an existing melody exactly. "
        "Each line must have precisely the requested number of syllables. "
        "Stressed positions (X) must fall on naturally stressed syllables. "
        "Write in the requested language, transliterated into Roman script "
        "so it can be sung by a synthesiser. Every line must be different "
        "from the others. "
        'Return JSON only, in exactly this shape: {"lines": ["...", "..."]} '
        "- one string per numbered line, in order.")
    user = (
        f"Language: {brief.language}\n"
        f"Mood: {brief.mood}\nFeel: {brief.feel}\n"
        f"Situation: {brief.situation}\nNotes: {brief.notes}\n\n"
        f"Lines to write:\n" + "\n".join(lines))
    return system, user


def intent(text: str, intents: Sequence[str]) -> Prompt:
    system = ("You classify a music director's spoken instruction into one "
              "intent from a fixed list. Return JSON only: "
              '{"intent": "...", "confidence": 0.0-1.0, "instrument": "..."}. '
              "Use \"unknown\" if nothing fits.")
    user = f"Instruction: {text!r}\nAllowed intents: {', '.join(intents)}"
    return system, user


def raagas(brief: Any, candidates: Sequence[str]) -> Prompt:
    system = ("You are a Carnatic and Hindustani music adviser. Choose raagas "
              "from the supplied list only. Return JSON only, in exactly this "
              'shape: {"raagas": [{"raaga": "...", "reason": "one sentence"}]} '
              "- best first, three or four of them.")
    user = (f"Brief: mood={brief.mood}; feel={brief.feel}; "
            f"situation={brief.situation}; language={brief.language}\n"
            f"Available raagas: {', '.join(candidates)}")
    return system, user


def instruments(description: str, catalog: Sequence[str]) -> Prompt:
    system = ("Choose instruments for a described feel. Pick only from the "
              "supplied list. Return JSON only, in exactly this shape: "
              '{"instruments": ["key", "key"]} - best first, at most four.')
    user = f"Feel: {description}\nAvailable: {', '.join(catalog)}"
    return system, user


def explain(question: str, context: str = "") -> Prompt:
    system = ("You are the arranger sitting beside a music director. Answer "
              "in at most three sentences, practically.")
    return system, (f"{context}\n\n{question}" if context else question)


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S)


def extract_json(text: str) -> Optional[Any]:
    """Pull a JSON value out of a model's reply.

    Tolerates a fenced code block, a preamble sentence, and trailing chatter -
    all of which small local models produce even when told not to.  Returns
    ``None`` rather than raising, because every caller has a working fallback.
    """
    if not text:
        return None
    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:                                        # noqa: BLE001
            pass
    try:
        return json.loads(text.strip())
    except Exception:                                            # noqa: BLE001
        pass
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:                                            # noqa: BLE001
        return None


def _numbered(data: Dict[str, Any]) -> List[Any]:
    """Values of a ``{"1": ..., "2": ...}`` object, in numeric order.

    A model told to answer a numbered list, and constrained to emit a JSON
    object, very often numbers the keys instead of returning an array.  That
    is a reasonable reading of the request, so it is accepted.
    """
    if data and all(str(k).strip().isdigit() for k in data):
        return [data[k] for k in sorted(data, key=lambda k: int(str(k).strip()))]
    return []


def as_lyrics(data: Any) -> List[str]:
    if isinstance(data, list):
        return [str(x).strip() for x in data]
    if isinstance(data, dict):                   # {"lines": [...]} is common
        for key in ("lines", "lyrics", "result"):
            if isinstance(data.get(key), list):
                return [str(x).strip() for x in data[key]]
        numbered = _numbered(data)
        if numbered:
            return [str(x).strip() for x in numbered]
    return []


def as_intent(data: Any) -> Dict[str, Any]:
    return data if isinstance(data, dict) else {}


def as_raagas(data: Any) -> List[Dict[str, str]]:
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("raagas", "result", "suggestions"):
            if isinstance(data.get(key), list):
                return [d for d in data[key] if isinstance(d, dict)]
        # A single suggestion, unwrapped - one answer is still an answer.
        if "raaga" in data:
            return [data]
        return [d for d in _numbered(data) if isinstance(d, dict)]
    return []


def as_instruments(data: Any, catalog: Sequence[str]) -> List[str]:
    allowed = set(catalog)
    if isinstance(data, dict):
        for key in ("instruments", "result"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = _numbered(data) or data
    if isinstance(data, list):
        # An entry may be a bare key or {"instrument": "veena"}.
        out = []
        for item in data:
            key = item.get("instrument") if isinstance(item, dict) else item
            if str(key) in allowed:
                out.append(str(key))
        return out
    return []
