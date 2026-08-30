"""Lyric-to-melody fitting engine (spec section 4 step 4).

The tune is written first and the lyrics are fitted to it -- syllable count,
note duration, stress, phrase length and breath positions all come from the
approved melody, never the other way round.

A *slot* is one breath phrase of the tune: the notes between two rests.  Each
slot states exactly how many syllables it wants and which of them fall on
stressed (long or downbeat) notes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.models import (LyricLine, LyricsVersion, MelodyVersion, Note,
                           Section)
from ..music.melody import phrase_boundaries
from ..music.theory import beat_seconds

VOWEL_GROUP = re.compile(r"(aa|ee|oo|ai|au|ae|ou|[aeiou])", re.I)


@dataclass
class PhraseSlot:
    section_id: str
    section_name: str
    note_indices: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    stresses: List[bool] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def syllable_count(self) -> int:
        return len(self.note_indices)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def describe(self) -> str:
        pattern = "".join("X" if s else "." for s in self.stresses)
        return (f"{self.section_name} {self.start:6.1f}s  "
                f"{self.syllable_count:2d} syllables  {pattern}")


def build_slots(melody: MelodyVersion, include_instrumental: bool = False
                ) -> List[PhraseSlot]:
    beat = beat_seconds(melody.tempo_bpm)
    slots: List[PhraseSlot] = []
    for section in melody.sections:
        if not include_instrumental and section.kind.instrumental:
            continue
        for group in phrase_boundaries(melody, section.id):
            if not group:
                continue
            notes = [melody.notes[i] for i in group]
            durations = [n.duration for n in notes]
            longest = max(durations)
            stresses = [d >= max(beat * 0.95, longest * 0.7) for d in durations]
            if stresses and not stresses[0]:
                stresses[0] = True     # phrase openings carry weight
            slots.append(PhraseSlot(
                section_id=section.id, section_name=section.name,
                note_indices=list(group), durations=durations, stresses=stresses,
                start=notes[0].start, end=notes[-1].end))
    return slots


def syllabify(word: str) -> List[str]:
    """Split a Roman-transliterated word into singable syllables."""
    w = word.strip()
    if not w:
        return []
    letters = re.sub(r"[^A-Za-z]", "", w)
    if not letters:
        return []
    syllables: List[str] = []
    i = 0
    current = ""
    while i < len(letters):
        m = VOWEL_GROUP.match(letters, i)
        if m:
            current += m.group(0)
            i = m.end()
            # A single trailing consonant closes the syllable.
            if i < len(letters) and not VOWEL_GROUP.match(letters, i):
                nxt = VOWEL_GROUP.search(letters, i)
                consonants = letters[i:nxt.start()] if nxt else letters[i:]
                if len(consonants) > 1 or not nxt:
                    current += consonants[0]
                    i += 1
            syllables.append(current)
            current = ""
        else:
            current += letters[i]
            i += 1
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return syllables


def count_syllables(text: str) -> int:
    return sum(len(syllabify(w)) for w in text.split())


def split_line_syllables(text: str) -> List[str]:
    out: List[str] = []
    for word in text.split():
        out.extend(syllabify(word))
    return out


def _vowel_of(syllable: str) -> str:
    m = VOWEL_GROUP.search(syllable or "")
    return (m.group(0).lower() if m else "a")


def fit_line(text: str, slot: PhraseSlot) -> Tuple[List[str], List[int], List[str]]:
    """Map a written line onto a slot's notes.

    Returns (syllables_per_note, note_indices, warnings).  Fewer syllables than
    notes produces melisma -- the vowel is carried across the extra notes.
    """
    warnings: List[str] = []
    syllables = split_line_syllables(text)
    notes = list(slot.note_indices)
    if not syllables:
        return [], notes, ["Line is empty."]

    if len(syllables) == len(notes):
        return syllables, notes, warnings

    if len(syllables) < len(notes):
        # Spread: hold the longest notes with melisma.
        out = list(syllables)
        extra = len(notes) - len(syllables)
        order = sorted(range(len(syllables)),
                       key=lambda i: -slot.durations[min(i, len(slot.durations) - 1)])
        holds: Dict[int, int] = {}
        for k in range(extra):
            idx = order[k % len(order)]
            holds[idx] = holds.get(idx, 0) + 1
        expanded: List[str] = []
        for i, syl in enumerate(out):
            expanded.append(syl)
            for _ in range(holds.get(i, 0)):
                expanded.append("~" + _vowel_of(syl))
        return expanded[:len(notes)], notes, warnings

    # More syllables than notes: pack the surplus onto the longest notes.
    warnings.append(
        f"{len(syllables)} syllables for {len(notes)} notes; "
        f"{len(syllables) - len(notes)} were doubled up.")
    packed: List[str] = [""] * len(notes)
    per_note = [1] * len(notes)
    surplus = len(syllables) - len(notes)
    order = sorted(range(len(notes)), key=lambda i: -slot.durations[i])
    for k in range(surplus):
        per_note[order[k % len(order)]] += 1
    pos = 0
    for i, count in enumerate(per_note):
        packed[i] = "".join(syllables[pos:pos + count])
        pos += count
    return packed, notes, warnings


def fit_lines(lines: Sequence[str], melody: MelodyVersion, language: str,
              version: int = 1,
              previous: Optional[LyricsVersion] = None) -> LyricsVersion:
    """Fit a list of written lines onto the melody's phrase slots."""
    slots = build_slots(melody)
    lv = LyricsVersion(version=version, language=language,
                       melody_version=melody.version)
    locked_by_slot: Dict[int, LyricLine] = {}
    if previous:
        for i, line in enumerate(previous.lines):
            if line.locked and i < len(slots):
                locked_by_slot[i] = line

    warnings: List[str] = []
    for i, slot in enumerate(slots):
        if i in locked_by_slot:
            lv.lines.append(locked_by_slot[i])
            continue
        text = lines[i] if i < len(lines) else ""
        syllables, indices, warn = fit_line(text, slot)
        warnings.extend(f"{slot.section_name}: {w}" for w in warn)
        lv.lines.append(LyricLine(
            section_id=slot.section_id, text=text, syllables=syllables,
            note_indices=indices, start=slot.start, end=slot.end))
    lv.notes = "\n".join(warnings)
    return lv


def refit_line(lyrics: LyricsVersion, melody: MelodyVersion, line_id: str,
               new_text: str) -> List[str]:
    """Re-fit one line in place, leaving every other line untouched."""
    line = lyrics.line_by_id(line_id)
    if line is None:
        raise KeyError(line_id)
    if line.locked:
        from ..core.versioning import LockedContentError
        raise LockedContentError("That line is locked.")
    slots = build_slots(melody)
    index = [l.id for l in lyrics.lines].index(line_id)
    if index >= len(slots):
        raise IndexError("The melody no longer has a phrase for that line.")
    syllables, indices, warnings = fit_line(new_text, slots[index])
    line.text = new_text
    line.syllables = syllables
    line.note_indices = indices
    line.start = slots[index].start
    line.end = slots[index].end
    return warnings


def alignment_report(lyrics: LyricsVersion, melody: MelodyVersion) -> str:
    slots = build_slots(melody)
    rows = []
    for i, line in enumerate(lyrics.lines):
        want = slots[i].syllable_count if i < len(slots) else 0
        have = len([s for s in line.syllables if not s.startswith("~")])
        flag = "ok" if want == len(line.syllables) else "MISFIT"
        rows.append(f"{line.start:7.1f}s  {want:2d} notes / {have:2d} syllables  "
                    f"{flag}  {line.text}")
    return "\n".join(rows)
