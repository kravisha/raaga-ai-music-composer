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


#: What a creator calls each section.  "pallavi" sits inside
#: "anupallavi", and the spoken spelling puts a space in the middle of it,
#: so these are matched longest first and each match is consumed - "anu
#: pallavi" must not also read as a request for a Pallavi.
SECTION_WORDS: Dict[SectionKind, Tuple[str, ...]] = {
    SectionKind.PRELUDE: ("prelude", "intro", "introduction"),
    SectionKind.PALLAVI: ("pallavi",),
    SectionKind.ANUPALLAVI: ("anupallavi", "anu pallavi", "anu-pallavi"),
    SectionKind.INTERLUDE: ("interlude",),
    SectionKind.CHARANAM: ("charanam", "charana"),
    SectionKind.BRIDGE: ("bridge",),
    SectionKind.OUTRO: ("outro", "ending", "coda"),
    SectionKind.VERSE: ("verse",),
    SectionKind.CHORUS: ("chorus",),
}

#: Longest first, so "anu pallavi" is read before "pallavi" is.
_ALIASES: Tuple[Tuple[SectionKind, str], ...] = tuple(sorted(
    ((kind, word) for kind, words in SECTION_WORDS.items() for word in words),
    key=lambda pair: -len(pair[1])))

#: What turns a name into a request.  A brief is mostly narrative, and a
#: narrative is full of these words used for something else, so a name on
#: its own is not an instruction.
_ASKING = ("include", "including", "add", "adds", "insert", "put in",
           "feature", "featuring", "contains", "containing", "structure",
           "sections", "start with", "starts with", "begin with",
           "begins with", "end with", "ends with", "open with", "close with")

#: And what turns it into a refusal.  Read close to the name, because
#: "no drums, and include a Charanam" refuses one thing and asks for
#: another in the same breath.
_REFUSING = ("no", "not", "without", "dont", "don't", "skip", "omit",
             "exclude", "avoid", "leave out", "drop")

#: How many words back a refusal still reaches.
_REFUSAL_REACH = 4


@dataclass(frozen=True)
class SectionRequest:
    """What the creator asked for, and what they asked against."""

    wanted: Tuple[SectionKind, ...] = ()
    refused: Tuple[SectionKind, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.wanted or self.refused)


def _clauses(blob: str) -> List[str]:
    return [c for c in re.split(r"[.;:\n!?]|,? but ", blob) if c.strip()]


def _named(clause: str) -> List[Tuple[SectionKind, int]]:
    """Every section named in one clause, with where it was named.

    Matches are consumed as they are found so a longer name cannot be
    read a second time as the shorter one inside it.
    """
    room = clause
    found = []
    for kind, word in _ALIASES:
        for match in re.finditer(rf"\b{re.escape(word)}\b", room):
            found.append((kind, match.start()))
        room = re.sub(rf"\b{re.escape(word)}\b",
                      lambda m: " " * len(m.group(0)), room)
    return found


def _refused_at(clause: str, at: int) -> bool:
    before = clause[:at].split()
    return any(w.strip(",") in _REFUSING for w in before[-_REFUSAL_REACH:])


def read_section_requests(*texts: str) -> SectionRequest:
    """The sections the creator actually asked for, or asked against.

    A name on its own is not a request: "a bridge between two worlds" is a
    description and "a happy ending to their long separation" is a story.
    A clause counts when it either carries a word that asks for something,
    or names two or more sections, which is how a creator writes a list of
    the structure they want.
    """
    blob = " ".join(t for t in texts if t).lower()
    if not blob:
        return SectionRequest()
    wanted: List[SectionKind] = []
    refused: List[SectionKind] = []
    for clause in _clauses(blob):
        names = _named(clause)
        if not names:
            continue
        asking = any(cue in clause for cue in _ASKING)
        listed = len({kind for kind, _ in names}) > 1
        if not (asking or listed):
            continue
        for kind, at in names:
            target = refused if _refused_at(clause, at) else wanted
            if kind not in target:
                target.append(kind)
    # A refusal wins: saying both is a contradiction, and the safer
    # reading of a contradiction is the one that leaves the song shorter
    # rather than the one that puts in something unwanted.
    wanted = [k for k in wanted if k not in refused]
    return SectionRequest(tuple(wanted), tuple(refused))


def sections_asked_for(*texts: str) -> Tuple[SectionKind, ...]:
    """Just the wanted half, for callers that do not care about refusals."""
    return read_section_requests(*texts).wanted


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
                  refused: Sequence[SectionKind] = (),
                  notes: Optional[List[str]] = None) -> List[Section]:
    """Lay out named sections that add up to roughly ``duration_target``.

    ``requested`` is what the creator asked for by name.  Those sections
    are not optional however short the song is: a template's optional
    repeat and an explicit request had been the same thing here, so a
    60-second brief that said "include an Anupallavi" came back without
    one and said nothing about it.  If one genuinely cannot fit, that goes
    into ``notes`` rather than happening quietly.

    ``refused`` is what they asked against, which a list of wanted names
    cannot express: "do not include an Anupallavi" is not silence about
    the Anupallavi.
    """
    slots = choose_template(song_type)
    cyc = cycle_seconds(tempo_bpm, beats_per_cycle)
    target = max(cyc * 4, float(duration_target or 150.0))
    asked = set(requested or ())
    unwanted = set(refused or ()) - asked
    said = notes if notes is not None else []

    # Asked against by name.  The template offering one is not a reason to
    # include it.
    if unwanted:
        kept = [s for s in slots if s.kind not in unwanted]
        if kept:
            slots = kept

    # Asked for by name, so no longer a candidate for pruning - but only
    # the first of each kind.  Asking for "an Interlude" is not asking for
    # both of the template's interludes, and protecting every repeat of a
    # requested kind kept a 60-second song at 67 seconds.
    for kind in asked:
        first = next((s for s in slots if s.kind is kind), None)
        if first is not None:
            first.optional = False

    # A creator who names the structure they want has not asked for the
    # template's reprises of it.  Those become the first things dropped
    # when the song is short - otherwise an unrequested Pallavi 2 and
    # Pallavi 3 stay mandatory and turn a request that fits comfortably
    # into an apology that it does not.
    if asked:
        seen = set()
        for slot in slots:
            if slot.kind in seen and slot.kind in asked:
                slot.optional = True
                slot.priority = 5
            seen.add(slot.kind)

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
        # Their number, not the clamped one.  ``target`` has already been
        # raised to four cycles, so quoting it told a creator who asked
        # for 30 seconds that they had asked for 31.
        wanted_seconds = float(duration_target or 150.0)
        said.append(
            f"{len(slots)} sections at one cycle each need "
            f"{floor:.0f}s, and you asked for about {wanted_seconds:.0f}s. "
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

    # Pruning a reprise leaves a hole in the numbering, and "Pallavi" then
    # "Pallavi 3" with no Pallavi 2 reads as a missing section rather than
    # a shorter song.  The numbers each kind already used are reassigned
    # in order to the slots that survived.  A locked section is addressed
    # by name, so its name is never one of the ones moved.
    by_kind: Dict[SectionKind, List[Slot]] = {}
    for slot in slots:
        by_kind.setdefault(slot.kind, []).append(slot)
    for kind, group in by_kind.items():
        if len(group) < 2:
            continue
        names = [s.name for s in choose_template(song_type)
                 if s.kind is kind][:len(group)]
        if len(names) < len(group) or any(n in locked for n in names):
            continue
        if any(s.name in locked for s in group):
            continue
        for slot, name in zip(group, names):
            slot.name = name
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
