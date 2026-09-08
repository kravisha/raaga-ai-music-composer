"""Where a Critic's revision points.

A revision is prose for a musician: "reshape the Anupallavi's opening",
"the passage at 9.93-11.49 s", "retain the Pallavi's motif".  This reads
the *place* out of it - the sections named, by any name the section parser
knows or by the tune's own labels, and the times given - and says which
sections can be rewritten, which are locked and left alone, which times
fall outside the song, and which revisions name no place at all.  It does
not read the musical property asked for; the engine has no such lever
yet, and nothing here pretends otherwise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ..music.structure import SECTION_WORDS

#: "9.93-11.49 s", "at 22.22–25.34 seconds", "from 4.8 to 9.3 s"
_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)?\s*(?:-|–|—|to)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)")
#: "at 14.89 s", "around 17.8 seconds"
_POINT = re.compile(r"(?:at|around|near|about)\s+(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds?\b)")


@dataclass
class Placement:
    """What a set of revisions asks to rewrite, placed on a tune."""
    targets: List = field(default_factory=list)        # Section objects, in time order
    skipped_locked: List[str] = field(default_factory=list)
    out_of_range: List[str] = field(default_factory=list)
    unplaced: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def placed_anything(self) -> bool:
        return bool(self.targets or self.skipped_locked or self.out_of_range)


def _sections_named(text: str, sections) -> List:
    """Sections a sentence names.  A numbered label ("Charanam 2") is exact;
    a bare kind word ("the Charanam") is the first section of that kind -
    the statement, not a reprise - which the note says."""
    covered = text.lower()
    found = []
    # Labels and kind words together, longest first and each consumed as
    # it matches, so "Anu Pallavi" is read before "Pallavi" can be found
    # inside it and "Charanam 2" before "Charanam".
    candidates = [(s.name.lower(), s) for s in sections]
    for kind, words in SECTION_WORDS.items():
        first = next((s for s in sections if s.kind == kind), None)
        if first is not None:
            candidates.extend((word, first) for word in words)
    for word, section in sorted(candidates, key=lambda pair: -len(pair[0])):
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, covered):
            if section not in found:
                found.append(section)
            covered = re.sub(pattern, lambda m: " " * len(m.group(0)), covered)
    return found


def _times_named(text: str) -> List[Tuple[float, float]]:
    spans = [(float(a), float(b)) for a, b in _RANGE.findall(text)]
    stripped = _RANGE.sub(" ", text)
    spans += [(float(t), float(t)) for t in _POINT.findall(stripped)]
    return [(min(a, b), max(a, b)) for a, b in spans]


def place_revisions(revisions: Sequence[str], melody) -> Placement:
    """Place each revision on the tune's sections."""
    placement = Placement()
    if melody is None or not melody.sections:
        placement.unplaced = list(revisions)
        return placement
    sections = list(melody.sections)
    song_end = max(s.end for s in sections)
    chosen: List = []
    for text in revisions:
        named = _sections_named(text, sections)
        spans = _times_named(text)
        for start, end in spans:
            if start > song_end + 0.01:
                placement.out_of_range.append(f"{start:g}-{end:g} s")
                continue
            for section in sections:
                if section.start < end + 1e-6 and section.end > start - 1e-6 \
                        and section not in named:
                    named.append(section)
        if not named and not spans:
            placement.unplaced.append(text)
            continue
        for section in named:
            if section.locked:
                if section.name not in placement.skipped_locked:
                    placement.skipped_locked.append(section.name)
            elif section not in chosen:
                chosen.append(section)
    placement.targets = sorted(chosen, key=lambda s: s.start)
    if placement.targets:
        placement.notes.append("rewriting " + ", ".join(s.name for s in placement.targets))
    if placement.skipped_locked:
        placement.notes.append(", ".join(placement.skipped_locked)
                               + " is locked and was left alone")
    if placement.out_of_range:
        placement.notes.append("the passage at " + "; ".join(placement.out_of_range)
                               + " is outside the song; nothing was rewritten for it")
    if placement.unplaced:
        placement.notes.append(f"{len(placement.unplaced)} revision(s) name no passage "
                               f"that could be placed")
    return placement
