"""Where a Critic's revision points.

A revision is prose for a musician: "reshape the Anupallavi's opening",
"the passage at 9.93-11.49 s", "keep the Pallavi unchanged; rewrite the
Charanam".  This reads the *place* out of it - the sections named, by any
name the section parser knows or by the tune's own labels, and the times
given - and says which sections are asked to change, which are asked to
be kept, which are locked and left alone, which times fall outside the
song, whether the whole tune was meant, and which revisions name a place
the reader cannot resolve.  It does not read the musical property asked
for; the engine has no such lever yet, and nothing here pretends
otherwise.

Three distinctions matter to the Producer, because each gets a different
answer:

* a *target* is rewritten; a section the Critic asked to keep ("retain
  the Pallavi's motif", "leave the Charanam alone", "the Pallavi works")
  is never one, whatever else the sentence says;
* the *whole tune* named - "the entire melody", "overall", or a remark
  that names no place at all, "make it warmer" - is a whole rewrite,
  sparing anything asked to be kept;
* a place named that cannot be found - "the phrase after the leap",
  "bar 3" - is *unresolved*: not a whole rewrite, not a guess.

Sections are half-open: a point exactly on a boundary belongs to the
section that starts there, and a range that ends on a boundary does not
reach into the next section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..music.structure import SECTION_WORDS

#: "9.93-11.49 s", "at 22.22–25.34 seconds", "from 4.8 to 9.3 s"
_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)?\s*(?:-|–|—|to)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)")
#: "at 14.89 s", "around 17.8 seconds"
_POINT = re.compile(r"(?:at|around|near|about)\s+(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)")

#: A clause is the unit of intent: "keep the Pallavi; rewrite the Charanam"
#: says two things.
_CLAUSE = re.compile(r"\s*(?:[;:]|(?<!\d)[.,](?!\d)|\bbut\b|\bwhile\b|\bwhereas\b|\bthen\b|"
                     r"\bexcept\b)\s*")

#: Words that ask for something to stay.  Read before or after the place
#: they govern: "keep the Pallavi", "the Pallavi unchanged", "the Pallavi
#: works".  "Leave" and "hold" are not here: "leave the singer room at the
#: Pallavi's cadence" and "hold the note at 14 s" ask for a change, and
#: "leave the Pallavi alone" is read from its "alone".
_PRESERVE = re.compile(
    r"\b(?:keep|keeps|kept|retain|retains|retained|preserve|"
    r"preserves|preserved|maintain|maintains|unchanged|untouched|"
    r"intact|alone|as is|as it is|as they are|as it was|stays?|remains?|works|"
    r"is fine|is good|is right|is strong|are fine|are good|"
    r"(?:don't|do not|never|without)\s+(?:change|changing|touch|touching|alter|"
    r"altering|rewrite|rewriting|vary|varying|move|moving))\b")

#: Words that ask for a change, so that "keep the Pallavi and rewrite the
#: Charanam" reads the Charanam as a target although "keep" came first.
_CHANGE = re.compile(
    r"\b(?:rewrite|reshape|revise|vary|develop|change|redo|rework|recompose|"
    r"regenerate|fix|tighten|strengthen|give|make|open|differentiate|smooth|"
    r"simplify|extend|shorten|alter|adjust|refine|replace|improve|reduce|raise|"
    r"lower|move|shift|add|remove|cut|try|bring|build|contrast|reconsider|"
    r"revisit|address|needs?|should|could|must|drags|sags|stalls|repeats|"
    r"wanders|clashes|sits|feels|sounds|lacks|misses)\b")

#: The whole tune, said outright.
_WHOLE = re.compile(
    r"\b(?:(?:whole|entire|complete|full)\s+(?:tune|song|melody|piece|"
    r"composition|thing|line)|overall|throughout|everywhere|from scratch|"
    r"start over|start again|as a whole|top to bottom|every section|"
    r"all sections|all the sections|all of it|everything|the tune|the melody|"
    r"the song)\b")

#: A place named by something other than a section or a time.  Present
#: without either, the revision points somewhere the reader cannot find,
#: and that is said rather than guessed.
_PLACE = re.compile(
    r"\b(?:passages?|phrases?|bars?|measures?|beats?|notes?|leaps?|cadences?|"
    r"openings?|endings?|beginnings?|close|closing|middle|sections?|parts?|"
    r"lines?|joins?|transitions?|cycles?|avartanam?s?|climax|peaks?|descents?|"
    r"ascents?|half|start|end|motifs?|motives?|figures?|runs?|turns?|"
    r"sangatis?|gamakas?|landings?|resolutions?|approach|swaras?|svaras?|"
    r"syllables?|words?|repeats?|reprises?|first|second|third|last|final)\b|"
    r"\b[srgmpdn]\d?[+'-]?(?:\s*[–—-]\s*[srgmpdn]\d?[+'-]?)+\b")

#: "the Pallavi's ending", "its opening", "the ending of the Charanam":
#: the part of a section, not the Outro or the Prelude.
_PART_OF = re.compile(
    r"((?:'s|\bits|\btheir)\s+)(ending|intro|introduction|coda|opening|close)\b")
_PART_OF_THE = re.compile(
    r"\b(ending|intro|introduction|coda|opening|close)(\s+of\s+(?:the\s+)?)")


@dataclass
class Placement:
    """What a set of revisions asks to rewrite, placed on a tune."""
    targets: List = field(default_factory=list)        # Section objects, in time order
    preserved: List[str] = field(default_factory=list)  # asked to be kept
    skipped_locked: List[str] = field(default_factory=list)
    out_of_range: List[str] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)   # a place named, not found
    whole: bool = False                                 # the whole tune, meant
    notes: List[str] = field(default_factory=list)
    #: The clause(s) that asked each target to change, by section id, so
    #: that what was asked of the Charanam is applied to the Charanam and
    #: not to another target named in the same breath.
    clauses: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def placed_anything(self) -> bool:
        return bool(self.targets or self.skipped_locked or self.out_of_range
                    or self.preserved or self.whole)


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _mentions(text: str, sections) -> Tuple[List[Tuple[int, int, object]], str]:
    """Every section a clause names, with where it was said.  A numbered
    label ("Charanam 2") is exact; a bare kind word ("the Charanam") is the
    first section of that kind - the statement, not a reprise.  Returns the
    mentions and the clause with them blanked out."""
    covered = text
    # "the Pallavi's ending" is the Pallavi; "the ending of the Pallavi" too.
    covered = _PART_OF.sub(lambda m: m.group(1) + " " * len(m.group(2)), covered)
    covered = _PART_OF_THE.sub(lambda m: " " * len(m.group(1)) + m.group(2), covered)
    found: List[Tuple[int, int, object]] = []
    # Labels and kind words together, longest first and each consumed as
    # it matches, so "Anu Pallavi" is read before "Pallavi" can be found
    # inside it and "Charanam 2" before "Charanam".
    candidates = [(s.name.lower(), s) for s in sections]
    for kind, words in SECTION_WORDS.items():
        first = next((s for s in sections if s.kind == kind), None)
        if first is not None:
            candidates.extend((word, first) for word in words)
    for word, section in sorted(candidates, key=lambda pair: -len(pair[0])):
        for match in list(re.finditer(rf"\b{re.escape(word)}\b", covered)):
            found.append((match.start(), match.end(), section))
            covered = _blank(covered, match.start(), match.end())
    found.sort(key=lambda item: item[0])
    return found, covered


def _spans(text: str) -> Tuple[List[Tuple[int, int, Tuple[float, float]]], str]:
    """Every time a clause names, with where it was said."""
    found = []
    covered = text
    for match in _RANGE.finditer(text):
        a, b = float(match.group(1)), float(match.group(2))
        found.append((match.start(), match.end(), (min(a, b), max(a, b))))
        covered = _blank(covered, match.start(), match.end())
    for match in _POINT.finditer(covered):
        t = float(match.group(1))
        found.append((match.start(), match.end(), (t, t)))
        covered = _blank(covered, match.start(), match.end())
    found.sort(key=lambda item: item[0])
    return found, covered


def _kept(clause: str, start: int, end: int, others: List[int]) -> bool:
    """Whether the place said at ``start:end`` is asked to stay.  The nearest
    cue before it decides - "keep the Pallavi and rewrite the Charanam" -
    and failing one, a cue after it up to the next place named: "the
    Pallavi unchanged", "the Pallavi works"."""
    before = clause[:start]
    last_keep = max((m.end() for m in _PRESERVE.finditer(before)), default=-1)
    last_change = max((m.end() for m in _CHANGE.finditer(before)), default=-1)
    if last_keep >= 0 and last_keep > last_change:
        return True
    if last_change >= 0:
        return False
    stop = min([p for p in others if p > start] + [len(clause)])
    after = clause[end:stop]
    keep = _PRESERVE.search(after)
    change = _CHANGE.search(after)
    return bool(keep) and (change is None or keep.start() < change.start())


def _sections_at(sections, start: float, end: float, song_end: float) -> List:
    """The sections a time span covers.  Half-open: a point on a boundary
    belongs to the section that starts there; a range ending on a boundary
    stops short of the next section; the song's end belongs to the last
    section."""
    if start == end:
        for section in sections:
            if section.start - 1e-6 <= start < section.end - 1e-6:
                return [section]
        return [sections[-1]] if start <= song_end + 0.01 else []
    return [s for s in sections
            if s.start < end - 1e-6 and s.end > start + 1e-6]


def place_revisions(revisions: Sequence[str], melody) -> Placement:
    """Place each revision on the tune's sections."""
    placement = Placement()
    if melody is None or not melody.sections:
        placement.unplaced = list(revisions)
        return placement
    sections = list(melody.sections)
    song_end = max(s.end for s in sections)
    chosen: List = []
    kept: List = []
    whole_asked = False
    whole_kept = False
    for text in revisions:
        found_place = False
        for clause in _CLAUSE.split(text.lower()):
            if not clause.strip():
                continue
            named, rest = _mentions(clause, sections)
            spans, rest = _spans(rest)
            positions = [m[0] for m in named] + [m[0] for m in spans]
            for start, end, section in named:
                found_place = True
                if _kept(clause, start, end, positions):
                    if section not in kept:
                        kept.append(section)
                elif section.locked:
                    if section.name not in placement.skipped_locked:
                        placement.skipped_locked.append(section.name)
                else:
                    if section not in chosen:
                        chosen.append(section)
                    placement.clauses.setdefault(section.id, []).append(clause.strip())
            for start, end, (t0, t1) in spans:
                found_place = True
                if t0 > song_end + 0.01:
                    placement.out_of_range.append(f"{t0:g}-{t1:g} s")
                    continue
                keep_it = _kept(clause, start, end, positions)
                for section in _sections_at(sections, t0, t1, song_end):
                    if keep_it:
                        if section not in kept:
                            kept.append(section)
                    elif section.locked:
                        if section.name not in placement.skipped_locked:
                            placement.skipped_locked.append(section.name)
                    else:
                        if section not in chosen:
                            chosen.append(section)
                        placement.clauses.setdefault(section.id, []).append(clause.strip())
            whole = _WHOLE.search(rest)
            if whole:
                found_place = True
                if _kept(clause, whole.start(), whole.end(), positions):
                    whole_kept = True
                else:
                    whole_asked = True
                continue
            if not named and not spans:
                if _PLACE.search(rest):
                    # A place is named that is neither a section nor a time.
                    placement.unplaced.append(text)
                    found_place = True
                    break
        if not found_place:
            # Nothing placed and no place named: a remark about the tune.
            whole_asked = True
    kept_names = [s.name for s in kept]
    if whole_kept and not whole_asked:
        placement.preserved.append("the whole tune")
    placement.preserved.extend(n for n in kept_names if n not in placement.preserved)
    chosen = [s for s in chosen if s not in kept]
    if whole_asked and not chosen and not whole_kept:
        if kept:
            chosen = [s for s in sections if not s.locked and s not in kept]
            for s in chosen:
                placement.clauses.setdefault(s.id, []).extend(revisions)
            placement.notes.append("the whole tune was asked for; every section but "
                                   + ", ".join(kept_names) + " is rewritten")
        else:
            placement.whole = True
    placement.targets = sorted(chosen, key=lambda s: s.start)
    if placement.targets:
        placement.notes.append("rewriting " + ", ".join(s.name for s in placement.targets))
    if placement.preserved:
        placement.notes.append(", ".join(placement.preserved)
                               + " kept, as the Critic asked")
    if placement.skipped_locked:
        placement.notes.append(", ".join(placement.skipped_locked)
                               + " is locked and was left alone")
    if placement.out_of_range:
        placement.notes.append("the passage at " + "; ".join(placement.out_of_range)
                               + " is outside the song; nothing was rewritten for it")
    if placement.unplaced:
        placement.notes.append(f"{len(placement.unplaced)} revision(s) name a passage "
                               f"that could not be placed: "
                               + " | ".join(t[:80] for t in placement.unplaced))
    if placement.whole:
        placement.notes.append("the whole tune is asked for")
    return placement
