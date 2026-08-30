"""Natural-language timeline parser (spec section 6).

Turns spoken time references into concrete ranges against the project:
absolute stamps, relative moves, ordinal minutes, named sections, the current
playhead and the selected region.

The rule the spec calls out explicitly is honoured here: "from the second
minute to the third minute" means 01:00-03:00 -- "from the Nth minute" starts
where that minute starts, and "to the Nth minute" runs to where it ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..core.models import Section

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90,
    "hundred": 100, "half": 0.5, "a": 1, "an": 1, "couple": 2,
}

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1, "final": -1,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7,
    "8th": 8, "9th": 9, "10th": 10,
}

SECTION_WORDS = {
    "prelude": ["prelude", "intro", "introduction", "opening"],
    "pallavi": ["pallavi", "chorus", "hook", "refrain"],
    "anupallavi": ["anupallavi"],
    "charanam": ["charanam", "verse", "stanza"],
    "interlude": ["interlude", "instrumental break", "break"],
    "bridge": ["bridge", "middle eight"],
    "outro": ["outro", "ending", "coda", "finish"],
}

END_TAIL_SECONDS = 30.0


@dataclass
class TimeContext:
    duration: float = 0.0
    playhead: float = 0.0
    selection: Optional[Tuple[float, float]] = None
    sections: List[Section] = field(default_factory=list)

    def section_at(self, t: float) -> Optional[Section]:
        return next((s for s in self.sections if s.start <= t < s.end), None)

    def find_section(self, word: str, ordinal: Optional[int] = None
                     ) -> Optional[Section]:
        word = (word or "").strip().lower()
        if not word:
            return None
        matches: List[Section] = []
        for s in self.sections:
            name = s.name.lower()
            kind = s.kind.value.lower()
            for canonical, spellings in SECTION_WORDS.items():
                if word in spellings and (canonical in kind or canonical in name):
                    matches.append(s)
                    break
            else:
                if word in name:
                    matches.append(s)
        if not matches:
            return None
        if ordinal is None:
            return matches[0]
        if ordinal == -1:
            return matches[-1]
        return matches[min(len(matches), ordinal) - 1]


@dataclass
class TimeSpec:
    start: Optional[float] = None
    end: Optional[float] = None
    relative: Optional[float] = None
    description: str = ""
    source: str = ""
    section_id: str = ""

    @property
    def is_range(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def is_point(self) -> bool:
        return self.start is not None and self.end is None

    def clamp(self, duration: float) -> "TimeSpec":
        if duration <= 0:
            return self
        if self.start is not None:
            self.start = max(0.0, min(self.start, duration))
        if self.end is not None:
            self.end = max(0.0, min(self.end, duration))
            if self.start is not None and self.end <= self.start:
                self.end = min(duration, self.start + 1.0)
        return self


def _word_number(text: str) -> Optional[float]:
    text = text.strip().lower().replace("-", " ")
    if not text:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    total = 0.0
    found = False
    for token in text.split():
        if token in NUMBER_WORDS:
            total += NUMBER_WORDS[token]
            found = True
        elif re.fullmatch(r"\d+", token):
            total += float(token)
            found = True
    return total if found else None


# The word boundaries are part of the pattern: without them a short number
# word such as "a" matches inside an ordinary word ("forw-a-rd") and steals
# the slot from the real number.
_NUM = (r"(?:\b(?:\d+(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r"|"
        + "|".join(ORDINALS) + r")\b)")


def _parse_clock(text: str) -> Optional[float]:
    m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _seconds_phrase(text: str) -> Optional[float]:
    m = re.search(rf"({_NUM}(?:\s+{_NUM})?)\s*(seconds?|secs?|s)\b", text)
    if m:
        return _word_number(m.group(1))
    return None


def _minutes_phrase(text: str) -> Optional[float]:
    m = re.search(rf"({_NUM}(?:\s+{_NUM})?)\s*(minutes?|mins?)\b", text)
    if m:
        v = _word_number(m.group(1))
        return v * 60.0 if v is not None else None
    return None


def _ordinal_in(text: str) -> Optional[int]:
    for word, value in ORDINALS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return None


def parse(text: str, ctx: TimeContext) -> Optional[TimeSpec]:
    """Parse a time reference out of a spoken sentence."""
    if not text:
        return None
    t = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    duration = max(0.0, ctx.duration)

    # --- explicit ranges: "from X to Y" -----------------------------------
    m = re.search(r"\bfrom\s+(.+?)\s+(?:to|till|until|through)\s+(.+?)(?:\.|$|\s*,)", t)
    if m:
        start = _parse_point(m.group(1), ctx, edge="start")
        end = _parse_point(m.group(2), ctx, edge="end")
        if start is not None or end is not None:
            spec = TimeSpec(start=start if start is not None else 0.0,
                            end=end if end is not None else duration,
                            description=f"{_fmt(start)} to {_fmt(end)}",
                            source="range")
            return spec.clamp(duration)

    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\.|$|\s*,)", t)
    if m:
        start = _parse_point(m.group(1), ctx, edge="start")
        end = _parse_point(m.group(2), ctx, edge="end")
        if start is not None and end is not None:
            return TimeSpec(start=start, end=end,
                            description=f"{_fmt(start)} to {_fmt(end)}",
                            source="range").clamp(duration)

    # --- "the first/last N seconds|minutes" -------------------------------
    m = re.search(rf"\b(first|last|final)\s+({_NUM}(?:\s+{_NUM})?)?\s*"
                  r"(seconds?|secs?|minutes?|mins?)\b", t)
    if m:
        which = m.group(1)
        amount = _word_number(m.group(2) or "one") or 1.0
        unit = m.group(3)
        span = amount * (60.0 if unit.startswith("min") else 1.0)
        if which == "first":
            return TimeSpec(0.0, span, description=f"first {_fmt(span)}",
                            source="first").clamp(duration)
        return TimeSpec(max(0.0, duration - span), duration,
                        description=f"last {_fmt(span)}",
                        source="last").clamp(duration)

    # --- "the second minute" (as a whole minute) --------------------------
    m = re.search(r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|"
                  r"eighth|ninth|tenth|last)\s+minute\b", t)
    if m:
        word = m.group(1)
        if word == "last":
            return TimeSpec(max(0.0, duration - 60.0), duration,
                            description="the last minute",
                            source="ordinal-minute").clamp(duration)
        n = ORDINALS[word]
        return TimeSpec((n - 1) * 60.0, n * 60.0,
                        description=f"the {word} minute",
                        source="ordinal-minute").clamp(duration)

    # --- named sections ---------------------------------------------------
    section, phrase = _find_section_phrase(t, ctx)
    if section is not None:
        if re.search(r"\b(from|starting at|start at)\b", t):
            return TimeSpec(section.start, None,
                            description=f"from the {section.name}",
                            source="section", section_id=section.id).clamp(duration)
        if re.search(r"\bafter\b", t):
            return TimeSpec(section.end, None,
                            description=f"after the {section.name}",
                            source="section", section_id=section.id).clamp(duration)
        if re.search(r"\bbefore\b", t):
            return TimeSpec(max(0.0, section.start), None,
                            description=f"before the {section.name}",
                            source="section", section_id=section.id).clamp(duration)
        return TimeSpec(section.start, section.end,
                        description=f"the {section.name}",
                        source="section", section_id=section.id).clamp(duration)

    # --- relative moves ---------------------------------------------------
    m = re.search(rf"\b(back|backward|backwards|rewind|earlier)\b.*?({_NUM}(?:\s+{_NUM})?)"
                  r"\s*(seconds?|secs?|minutes?|mins?)?", t)
    if m:
        amount = _word_number(m.group(2)) or 10.0
        if (m.group(3) or "").startswith("min"):
            amount *= 60
        return TimeSpec(relative=-amount, description=f"back {_fmt(amount)}",
                        source="relative")
    m = re.search(rf"\b(forward|forwards|ahead|skip|later)\b.*?({_NUM}(?:\s+{_NUM})?)"
                  r"\s*(seconds?|secs?|minutes?|mins?)?", t)
    if m:
        amount = _word_number(m.group(2)) or 10.0
        if (m.group(3) or "").startswith("min"):
            amount *= 60
        return TimeSpec(relative=amount, description=f"forward {_fmt(amount)}",
                        source="relative")

    # --- "five seconds before this point" ---------------------------------
    m = re.search(rf"({_NUM}(?:\s+{_NUM})?)\s*(seconds?|secs?|minutes?|mins?)\s+"
                  r"(before|after)\s+(this|that|here|there|the playhead|now)", t)
    if m:
        amount = _word_number(m.group(1)) or 5.0
        if m.group(2).startswith("min"):
            amount *= 60
        sign = -1 if m.group(3) == "before" else 1
        point = ctx.playhead + sign * amount
        return TimeSpec(max(0.0, point), None,
                        description=f"{_fmt(amount)} {m.group(3)} the playhead",
                        source="playhead-relative").clamp(duration)

    # --- whole-song and edges ---------------------------------------------
    if re.search(r"\b(the whole|whole song|entire song|from the (very )?start|"
                 r"from the beginning|all of it|everything)\b", t):
        return TimeSpec(0.0, duration, description="the whole song",
                        source="whole").clamp(duration)
    if re.search(r"\b(the end|at the end|ending|the finish|the last bit)\b", t):
        span = min(END_TAIL_SECONDS, max(5.0, duration * 0.25)) if duration else \
            END_TAIL_SECONDS
        return TimeSpec(max(0.0, duration - span), duration,
                        description="the end", source="end").clamp(duration)
    if re.search(r"\b(the beginning|the start|the top)\b", t):
        return TimeSpec(0.0, min(duration or 30.0, 30.0),
                        description="the beginning", source="start").clamp(duration)

    # --- selection and playhead ------------------------------------------
    if re.search(r"\b(this (part|bit|section|region)|the selection|selected|"
                 r"this range)\b", t):
        if ctx.selection:
            s, e = ctx.selection
            return TimeSpec(s, e, description="the selected range",
                            source="selection").clamp(duration)
        section = ctx.section_at(ctx.playhead)
        if section:
            return TimeSpec(section.start, section.end,
                            description=f"the {section.name}", source="section",
                            section_id=section.id).clamp(duration)
    if re.search(r"\b(here|there|at this point|right now|from now)\b", t):
        return TimeSpec(ctx.playhead, None, description="here",
                        source="playhead").clamp(duration)

    # --- bare clock or unit phrases ---------------------------------------
    clock = _parse_clock(t)
    if clock is not None:
        if re.search(r"\b(from|at|start(ing)? (at|from))\b", t):
            return TimeSpec(clock, None, description=_fmt(clock),
                            source="clock").clamp(duration)
        return TimeSpec(clock, None, description=_fmt(clock),
                        source="clock").clamp(duration)
    secs = _seconds_phrase(t)
    mins = _minutes_phrase(t)
    value = None
    if mins is not None:
        value = mins + (secs or 0.0)
    elif secs is not None:
        value = secs
    if value is not None:
        if re.search(r"\bfor\b", t):
            return TimeSpec(ctx.playhead, ctx.playhead + value,
                            description=f"{_fmt(value)} from here",
                            source="duration").clamp(duration)
        return TimeSpec(value, None, description=_fmt(value),
                        source="absolute").clamp(duration)
    return None


def _parse_point(fragment: str, ctx: TimeContext, edge: str) -> Optional[float]:
    """Resolve one side of a 'from X to Y' expression."""
    f = " " + fragment.strip() + " "
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
                  r"ninth|tenth|last)\s+minute\b", f)
    if m:
        word = m.group(1)
        if word == "last":
            return max(0.0, ctx.duration - 60.0) if edge == "start" else ctx.duration
        n = ORDINALS[word]
        return (n - 1) * 60.0 if edge == "start" else n * 60.0
    clock = _parse_clock(f)
    if clock is not None:
        return clock
    section, _ = _find_section_phrase(f, ctx)
    if section is not None:
        return section.start if edge == "start" else section.end
    if re.search(r"\b(the end|end)\b", f):
        return ctx.duration
    if re.search(r"\b(the (start|beginning)|start|beginning)\b", f):
        return 0.0
    if re.search(r"\b(here|this point|now)\b", f):
        return ctx.playhead
    mins = _minutes_phrase(f)
    secs = _seconds_phrase(f)
    if mins is not None or secs is not None:
        return (mins or 0.0) + (secs or 0.0)
    bare = _word_number(f)
    if bare is not None and re.search(r"\bminute", f):
        return bare * 60.0
    if bare is not None:
        return bare
    return None


def _find_section_phrase(text: str, ctx: TimeContext
                         ) -> Tuple[Optional[Section], str]:
    ordinal = None
    m = re.search(r"\b(first|second|third|fourth|fifth|last)\s+(\w+)", text)
    if m and m.group(2) not in ("minute", "minutes", "second", "seconds"):
        ordinal = ORDINALS.get(m.group(1))
    for canonical, spellings in SECTION_WORDS.items():
        for word in spellings:
            if re.search(rf"\b{re.escape(word)}\b", text):
                section = ctx.find_section(word, ordinal)
                if section is not None:
                    return section, word
    for s in ctx.sections:
        if re.search(rf"\b{re.escape(s.name.lower())}\b", text):
            return s, s.name
    return None, ""


def _fmt(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f}s"


def describe(spec: Optional[TimeSpec]) -> str:
    if spec is None:
        return "no time reference"
    if spec.relative is not None:
        return spec.description
    if spec.is_range:
        return f"{_fmt(spec.start)} - {_fmt(spec.end)}"
    if spec.is_point:
        return f"from {_fmt(spec.start)}"
    return spec.description or "no time reference"
