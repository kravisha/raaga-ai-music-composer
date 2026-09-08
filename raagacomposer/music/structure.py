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
           "begins with", "end with", "ends with", "open with", "close with",
           "keep", "i want", "we want", "i'd like", "we'd like",
           "i would like", "we would like", "i need", "we need")

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
    # A new explicit request after a comma or "and" starts a new scope.
    # Keep ordinary section-list commas together: "include Pallavi,
    # Anupallavi and Charanam" still applies one request to the list.
    asking = "|".join(re.escape(cue) for cue in _ASKING)
    boundary = rf"(?:,\s*|\s+and\s+)(?=(?:{asking})\b)"
    return [c for c in re.split(r"[.;:\n!?]|,? but |" + boundary, blob)
            if c.strip()]


#: Words that may sit between a refusal and the section it refuses.
_DETERMINERS = frozenset({"the", "a", "an", "any", "that", "this", "its",
                          "my", "our", "another", "second", "extra"})

#: Words a direction may carry *after* the section it names.  These say
#: which song or which take the instruction applies to - they qualify the
#: instruction rather than continuing a sentence about people.
_SCOPE_WORDS = frozenset({
    "in", "for", "on", "of", "at", "to",
    "this", "that", "the", "a", "an", "it", "one",
    "song", "songs", "version", "tune", "take", "draft", "section",
    "time", "now", "moment", "round", "pass", "attempt", "here",
    "again", "today", "tonight", "yet", "ever", "all",
    "please", "thanks", "thank", "you", "ok", "okay",
})

#: Filler that may open a direction before the refusal itself.
_POLITE = frozenset({"please", "do", "just", "and", "so", "also"})


def _starts_with_refusal(clause: str) -> bool:
    # A standalone exclusion is a request in its own right. Require a
    # directive at the start; "there is no bridge over the river" remains
    # a narrative rather than an instruction to remove a song section.
    cues = "|".join(re.escape(cue) for cue in _REFUSING)
    return bool(re.match(rf"\s*(?:please\s+)?(?:do\s+)?(?:{cues})\b",
                         clause))


def _reads_as_a_direction(clause: str,
                          names: List[Tuple[SectionKind, int]]) -> bool:
    """Does a refusal-opening clause have the shape of an instruction?

    Opening with a refusal is not enough by itself.  "No one told him
    about the interlude of his life" opens with one and is a story, and
    letting the whole clause count made it ask *for* an Interlude - the
    very fault this branch was added to fix, reappearing inside the fix.

    Counting the words after the name was not enough either.  "No
    Anupallavi in this song" and "Skip the Anupallavi for this version"
    are ordinary instructions that carry their scope with them, and a
    fixed budget threw them away - while "Anu Pallavi" spent part of that
    budget on its own second word, so the same direction behaved
    differently depending on how the creator spelled the section.

    The shape is the test instead: a refusal, then the section it refuses,
    then nothing but words that say which song or which take.  What
    follows "not the ending" is "they hoped for", and no arrangement of
    those words says which take anything applies to.
    """
    if not names:
        return False
    words = [w.strip(".,!?;:") for w in clause.split()]
    words = [w for w in words if w]
    i = 0
    while i < len(words) and words[i] in _POLITE:
        i += 1
    cue = next((c for c in sorted(_REFUSING, key=lambda c: -len(c.split()))
                if words[i:i + len(c.split())] == c.split()), None)
    if cue is None:
        return False
    i += len(cue.split())
    while i < len(words) and words[i] in _DETERMINERS:
        i += 1
    rest = " ".join(words[i:])
    alias = next((a for _, a in _ALIASES
                  if re.match(rf"{re.escape(a)}\b", rest)), None)
    if alias is None:
        return False
    i += len(alias.split())
    return all(w in _SCOPE_WORDS for w in words[i:])


def _named(clause: str) -> List[Tuple[SectionKind, int, int]]:
    """Every section named in one clause, and where the name begins and ends.

    Matches are consumed as they are found so a longer name cannot be
    read a second time as the shorter one inside it.  The end is carried
    because what follows the name is how a direction is told from a
    sentence, and "anu pallavi" ends two words after it starts.
    """
    room = clause
    found = []
    for kind, word in _ALIASES:
        for match in re.finditer(rf"\b{re.escape(word)}\b", room):
            found.append((kind, match.start(), match.end()))
        room = re.sub(rf"\b{re.escape(word)}\b",
                      lambda m: " " * len(m.group(0)), room)
    return sorted(found, key=lambda item: item[1])


def _refused_at(clause: str, at: int) -> bool:
    before = clause[:at].split()
    nearby = " ".join(w.strip(",") for w in before[-_REFUSAL_REACH:])
    return any(re.search(rf"\b{re.escape(cue)}\b", nearby)
               for cue in _REFUSING)


#: Words that may sit between an asking cue and the section it asks for.
_LEAD_IN = _DETERMINERS | {"in", "with", "to", "of", "on", "for", "at",
                           "into", "and", "also", "then"}

#: What may separate one name from the next in a list of sections.
_LIST_JOINERS = _DETERMINERS | {"and", "then", "plus", "&"}


def _tail_is_scope(clause: str, names: List[Tuple[SectionKind, int, int]]
                   ) -> bool:
    """After the last section named, is there anything but scope left?

    This is the same test the refusing side uses, and for the same
    reason: "in this song" says which song, and "of his story" carries on
    being a story.
    """
    if not names:
        return False
    end = max(finish for _, _, finish in names)
    rest = [w.strip(".,!?;:") for w in clause[end:].split()]
    return all(w in _SCOPE_WORDS for w in rest if w)


def _asked_for_at(clause: str, at: int) -> bool:
    """Does an asking cue actually govern the section named at *at*?

    A cue was matched anywhere in the clause, so any narrative carrying
    one of these ordinary words turned into an instruction: "his life
    contains a bridge he cannot cross" asked for a Bridge, and
    "he begins with a prelude of doubt" asked for a Prelude.  The cue has
    to be the thing introducing this name - determiners and a preposition
    may sit between them, a sentence about someone's life may not.

    The cue is looked for before each lead-in word is set aside, not only
    after all of them are.  Half the cues end in one - "begin with",
    "put in" - and stripping first ate the cue's own last word, so
    "begin with a prelude" asked for nothing.
    """
    before = [w.strip(".,!?;:") for w in clause[:at].split()]
    before = [w for w in before if w]
    while before:
        if any(before[-len(cue.split()):] == cue.split()
               for cue in _ASKING if len(cue.split()) <= len(before)):
            return True
        if before[-1] not in _LEAD_IN:
            return False
        before.pop()
    return False


def _reads_as_a_list(clause: str, names: List[Tuple[SectionKind, int, int]]
                     ) -> bool:
    """Are these names written out as a list, one after another?

    Naming two sections was enough by itself, so "a bridge between two
    worlds and a happy ending" counted as a list of two.  A list has
    nothing between its items but the words that join a list.
    """
    if len({kind for kind, _, _ in names}) < 2:
        return False
    ordered = sorted(names, key=lambda item: item[1])
    for (_, _, finish), (_, start, _) in zip(ordered, ordered[1:]):
        between = [w.strip(".,!?;:") for w in clause[finish:start].split()]
        if any(w and w not in _LIST_JOINERS for w in between):
            return False
    return True


def read_section_requests(*texts: str) -> SectionRequest:
    """The sections the creator actually asked for, or asked against.

    A name on its own is not a request: "a bridge between two worlds" is a
    description and "a happy ending to their long separation" is a story.
    A name counts three ways, and each of them has to be earned by the
    shape of the clause rather than by a word appearing somewhere in it:
    a cue that asks for *this* name, a list of sections written out one
    after another, or a refusal that stops where an instruction stops.
    In every case what follows the last name must say which song or which
    take, not carry on being a sentence.
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
        scoped = _tail_is_scope(clause, names)
        listed = scoped and _reads_as_a_list(clause, names)
        refusing = (_starts_with_refusal(clause)
                    and _reads_as_a_direction(clause, names))
        for kind, at, _ in names:
            denied = _refused_at(clause, at)
            asked = scoped and _asked_for_at(clause, at)
            if denied:
                # A refusal still has to be part of an instruction: it is
                # either a direction in its own right, or it sits inside
                # one - "do not include an Anupallavi".
                if not (refusing or listed or asked):
                    continue
                target = refused
            else:
                # And a name nothing asks for is not requested by being
                # in the same sentence as one that is.
                if not (listed or asked):
                    continue
                target = wanted
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
