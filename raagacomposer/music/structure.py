"""Song structure planning (spec section 9).

Produces the named, independently editable regions the creator directs by
name: prelude, pallavi, anupallavi, interlude 1, charanam, interlude 2,
bridge, outro.  Section boundaries are whole tala cycles so that arrangement
edits, regeneration and playback all land musically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.models import Section, SectionKind
from .theory import cycle_seconds


@dataclass
class Slot:
    kind: SectionKind
    name: str
    cycles: int
    intensity: float
    optional: bool = False
    priority: int = 0          # higher is dropped first when time is short


def _film_template() -> List[Slot]:
    return [
        Slot(SectionKind.PRELUDE, "Prelude", 2, 0.45),
        Slot(SectionKind.PALLAVI, "Pallavi", 4, 0.70),
        Slot(SectionKind.ANUPALLAVI, "Anupallavi", 3, 0.65, optional=True, priority=3),
        Slot(SectionKind.INTERLUDE, "Interlude 1", 2, 0.55),
        Slot(SectionKind.CHARANAM, "Charanam 1", 4, 0.65),
        Slot(SectionKind.PALLAVI, "Pallavi 2", 3, 0.75),
        Slot(SectionKind.INTERLUDE, "Interlude 2", 2, 0.60, optional=True, priority=2),
        Slot(SectionKind.CHARANAM, "Charanam 2", 4, 0.70, optional=True, priority=1),
        Slot(SectionKind.BRIDGE, "Bridge", 2, 0.50, optional=True, priority=4),
        Slot(SectionKind.PALLAVI, "Pallavi 3", 3, 0.80),
        Slot(SectionKind.OUTRO, "Outro", 2, 0.40),
    ]


def _simple_template() -> List[Slot]:
    return [
        Slot(SectionKind.PRELUDE, "Prelude", 1, 0.40),
        Slot(SectionKind.VERSE, "Verse 1", 3, 0.60),
        Slot(SectionKind.CHORUS, "Chorus 1", 3, 0.75),
        Slot(SectionKind.INTERLUDE, "Interlude 1", 2, 0.55),
        Slot(SectionKind.VERSE, "Verse 2", 3, 0.60, optional=True, priority=2),
        Slot(SectionKind.CHORUS, "Chorus 2", 3, 0.80),
        Slot(SectionKind.OUTRO, "Outro", 1, 0.35),
    ]


def _devotional_template() -> List[Slot]:
    return [
        Slot(SectionKind.PRELUDE, "Prelude", 2, 0.35),
        Slot(SectionKind.PALLAVI, "Pallavi", 4, 0.60),
        Slot(SectionKind.INTERLUDE, "Interlude 1", 2, 0.45),
        Slot(SectionKind.CHARANAM, "Charanam 1", 4, 0.60),
        Slot(SectionKind.PALLAVI, "Pallavi 2", 3, 0.65),
        Slot(SectionKind.CHARANAM, "Charanam 2", 4, 0.60, optional=True, priority=1),
        Slot(SectionKind.OUTRO, "Outro", 2, 0.35),
    ]


TEMPLATES = {
    "film song": _film_template,
    "film": _film_template,
    "devotional": _devotional_template,
    "bhajan": _devotional_template,
    "simple": _simple_template,
    "pop": _simple_template,
    "ghazal": _simple_template,
}


#: What a creator calls each section when they ask for one.  "pallavi"
#: is a part of "anupallavi", which is why these are matched on word
#: boundaries and not as substrings - asking for an Anupallavi must not
#: also read as asking for a Pallavi, and vice versa.
SECTION_WORDS: Dict[SectionKind, Tuple[str, ...]] = {
    SectionKind.PRELUDE: ("prelude", "intro", "introduction"),
    SectionKind.PALLAVI: ("pallavi",),
    SectionKind.ANUPALLAVI: ("anupallavi",),
    SectionKind.INTERLUDE: ("interlude",),
    SectionKind.CHARANAM: ("charanam", "charana"),
    SectionKind.BRIDGE: ("bridge",),
    SectionKind.OUTRO: ("outro", "ending", "coda"),
    SectionKind.VERSE: ("verse",),
    SectionKind.CHORUS: ("chorus",),
}


def sections_asked_for(*texts: str) -> Tuple[SectionKind, ...]:
    """The sections named in the creator's own words.

    Only a name counts.  This does not try to read intent from a mood or a
    situation - "a bridge between two worlds" is not a request for a
    Bridge section, and guessing would be worse than not looking.
    """
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return ()
    found = []
    for kind, words in SECTION_WORDS.items():
        for word in words:
            if re.search(rf"\b{re.escape(word)}\b", blob):
                found.append(kind)
                break
    return tuple(found)


def choose_template(song_type: str) -> List[Slot]:
    key = (song_type or "").strip().lower()
    for name, fn in TEMPLATES.items():
        if name in key:
            return fn()
    return _film_template()


def plan_sections(duration_target: float, tempo_bpm: int, beats_per_cycle: int,
                  song_type: str = "film song",
                  existing: Optional[List[Section]] = None,
                  requested: Sequence[SectionKind] = (),
                  notes: Optional[List[str]] = None) -> List[Section]:
    """Lay out named sections that add up to roughly ``duration_target``.

    ``requested`` is what the creator asked for by name.  Those sections
    are not optional however short the song is: a template's optional
    repeat and an explicit request had been the same thing here, so a
    60-second brief that said "include an Anupallavi" came back without
    one and said nothing about it.  If one genuinely cannot fit, that goes
    into ``notes`` rather than happening quietly.
    """
    slots = choose_template(song_type)
    cyc = cycle_seconds(tempo_bpm, beats_per_cycle)
    target = max(cyc * 4, float(duration_target or 150.0))
    asked = set(requested or ())
    said = notes if notes is not None else []

    # Asked for by name, so no longer a candidate for pruning - but only
    # the first of each kind.  Asking for "an Interlude" is not asking for
    # both of the template's interludes, and protecting every repeat of a
    # requested kind kept a 60-second song at 67 seconds.
    for kind in asked:
        first = next((s for s in slots if s.kind is kind), None)
        if first is not None:
            first.optional = False

    # Asked for and not in this template at all: add it before the ending,
    # which is where another one of its kind would have sat.
    have = {slot.kind for slot in slots}
    for kind in asked - have:
        at = next((i for i, s in enumerate(slots)
                   if s.kind is SectionKind.OUTRO), len(slots))
        slots.insert(at, Slot(kind, kind.value.title(), 3, 0.6))
        said.append(f"{kind.value.title()} is not usual in a "
                    f"{song_type or 'film song'}, so I have put one in "
                    f"because you asked for it.")

    def total(items: List[Slot]) -> float:
        return sum(s.cycles for s in items) * cyc

    # Drop optional sections while clearly over target.
    while total(slots) > target * 1.15:
        droppable = [s for s in slots if s.optional]
        if not droppable:
            break
        victim = max(droppable, key=lambda s: s.priority)
        slots.remove(victim)

    # Everything left is either structural or asked for, and it still does
    # not fit.  The sections are about to be squeezed to a cycle each, so
    # say that plainly instead of returning a song that quietly disagrees
    # with the brief.
    floor = len(slots) * cyc
    if asked and floor > target * 1.15:
        said.append(
            f"{len(slots)} sections at one cycle each need "
            f"{floor:.0f}s, and you asked for about {target:.0f}s. "
            f"I have kept every section you named and they are each as "
            f"short as a cycle allows.")

    # Scale remaining cycle counts toward the target.
    if total(slots) > 0:
        factor = target / total(slots)
        for s in slots:
            s.cycles = max(1, int(round(s.cycles * factor)))

    # Fine adjustment: add or remove single cycles from the biggest sections.
    guard = 0
    while abs(total(slots) - target) > cyc * 0.75 and guard < 64:
        guard += 1
        if total(slots) < target:
            s = max(slots, key=lambda s: (s.intensity, s.cycles))
            s.cycles += 1
        else:
            candidates = [s for s in slots if s.cycles > 1]
            if not candidates:
                break
            s = min(candidates, key=lambda s: (s.intensity, -s.cycles))
            s.cycles -= 1

    locked = {s.name: s for s in (existing or []) if s.locked}
    sections: List[Section] = []
    t = 0.0
    for slot in slots:
        dur = slot.cycles * cyc
        if slot.name in locked:
            old = locked[slot.name]
            sec = Section(id=old.id, name=old.name, kind=old.kind, start=t,
                          end=t + old.duration, locked=True,
                          intensity=old.intensity)
            t += old.duration
        else:
            sec = Section(name=slot.name, kind=slot.kind, start=t, end=t + dur,
                          intensity=slot.intensity)
            t += dur
        sections.append(sec)
    return sections


def section_role(kind: SectionKind) -> str:
    if kind in (SectionKind.PRELUDE, SectionKind.INTERLUDE, SectionKind.BRIDGE,
                SectionKind.OUTRO):
        return "instrumental"
    if kind in (SectionKind.PALLAVI, SectionKind.CHORUS):
        return "hook"
    return "verse"


def describe(sections: List[Section]) -> str:
    lines = []
    for s in sections:
        lines.append(f"{s.start:7.1f}s - {s.end:7.1f}s  {s.name:<14} "
                     f"{s.kind.value:<11}{' [locked]' if s.locked else ''}")
    return "\n".join(lines)
