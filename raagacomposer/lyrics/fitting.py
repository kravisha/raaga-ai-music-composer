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
import unicodedata
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


def _roman_letters(word: str) -> Tuple[str, List[str]]:
    """The word's letters as plain ASCII, beside the characters they came from.

    ``Praṇaṇa`` becomes ``Pranana`` for the split, and each plain letter
    remembers its written form (``ṇ``), so the syllables handed back carry
    the diacritics the creator wrote.  Anything that is not a letter -
    punctuation, digits - is dropped, as it always was.
    """
    skeleton: List[str] = []
    written: List[str] = []
    for ch in unicodedata.normalize("NFC", word):
        base = unicodedata.normalize("NFKD", ch)[0]
        if "A" <= base <= "Z" or "a" <= base <= "z":
            skeleton.append(base)
            written.append(ch)
        elif unicodedata.category(ch).startswith("M") and written:
            # A combining mark that survived NFC belongs to the letter before.
            written[-1] += ch
    return "".join(skeleton), written


def _syllabify_roman(word: str) -> List[str]:
    letters, written = _roman_letters(word)
    if not letters:
        return []
    spans: List[Tuple[int, int]] = []
    i = 0
    start = 0
    while i < len(letters):
        m = VOWEL_GROUP.match(letters, i)
        if m:
            i = m.end()
            # A single trailing consonant closes the syllable.
            if i < len(letters) and not VOWEL_GROUP.match(letters, i):
                nxt = VOWEL_GROUP.search(letters, i)
                consonants = letters[i:nxt.start()] if nxt else letters[i:]
                if len(consonants) > 1 or not nxt:
                    i += 1
            spans.append((start, i))
            start = i
        else:
            i += 1
    if start < len(letters):
        if spans:
            spans[-1] = (spans[-1][0], len(letters))
        else:
            spans.append((start, len(letters)))
    return ["".join(written[a:b]) for a, b in spans]


def _is_virama(ch: str) -> bool:
    try:
        return "VIRAMA" in unicodedata.name(ch)
    except ValueError:
        return False


def _syllabify_script(word: str) -> List[str]:
    """Syllables of a word in an Indic script, by its orthography.

    A base letter with the marks that follow it is one unit; a consonant
    killed by a virama (Tamil pulli) closes the unit before it, the way a
    single trailing consonant closes a Roman syllable.  ``திரும்பி`` is
    ``தி | ரும் | பி``.  This is orthographic syllabification, which for
    Tamil is what singers articulate; it is not a validated phonological
    analysis, and the fitter says so wherever it matters.
    """
    units: List[Tuple[str, bool]] = []
    for ch in unicodedata.normalize("NFC", word):
        category = unicodedata.category(ch)
        if category.startswith("M") or ch in ("‌", "‍"):
            if units:
                text, dead = units[-1]
                units[-1] = (text + ch, dead or _is_virama(ch))
        elif category.startswith("L"):
            units.append((ch, False))
    syllables: List[str] = []
    pending = ""
    for text, dead in units:
        if dead:
            if syllables:
                syllables[-1] += text
            else:
                pending += text      # a word-initial cluster: ஸ்ரீ
        else:
            syllables.append(pending + text)
            pending = ""
    if pending:
        if syllables:
            syllables[-1] += pending
        else:
            syllables.append(pending)
    return syllables


def _uses_a_script(word: str) -> bool:
    return any(not ch.isascii() and unicodedata.category(ch).startswith("L")
               and not unicodedata.normalize("NFKD", ch)[0].isascii()
               for ch in word)


def syllabify(word: str) -> List[str]:
    """Split one written word into singable syllables.

    Roman transliteration, with or without diacritics, splits by its
    vowels and keeps the letters as written.  A word in an Indic script
    splits by its orthography.  What comes back is the creator's text in
    pieces, never a re-spelling of it.
    """
    w = word.strip()
    if not w:
        return []
    if _uses_a_script(w):
        return _syllabify_script(w)
    return _syllabify_roman(w)


def count_syllables(text: str) -> int:
    return sum(len(syllabify(w)) for w in text.split())


def split_line_syllables(text: str) -> List[str]:
    out: List[str] = []
    for word in text.split():
        out.extend(syllabify(word))
    return out


def _vowel_of(syllable: str) -> str:
    """The vowel a melisma holds: the syllable's Roman vowel group, or in
    a script its last vowel sign or independent vowel."""
    m = VOWEL_GROUP.search(syllable or "")
    if m and syllable.isascii():
        return m.group(0).lower()
    signs = [ch for ch in syllable or "" if unicodedata.category(ch).startswith("M")
             and not _is_virama(ch)]
    if signs:
        return signs[-1]
    vowels = [ch for ch in syllable or "" if "VOWEL" in _name(ch)]
    if vowels:
        return vowels[-1]
    return m.group(0).lower() if m else "a"


def _name(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return ""


def fit_line(text: str, slot: PhraseSlot) -> Tuple[List[str], List[int], List[str]]:
    """Map a written line onto a slot's notes.

    Returns (syllables_per_note, note_indices, warnings).  Fewer syllables than
    notes produces melisma -- the vowel is carried across the extra notes.
    """
    warnings: List[str] = []
    syllables = split_line_syllables(text)
    notes = list(slot.note_indices)
    if not syllables:
        if text.strip():
            # Kept as written, given no notes: the words are the creator's
            # or the writer's, and a count is not a reason to replace them.
            return [], notes, [f"No singable syllables found in {text.strip()!r}; "
                               f"the line is kept as written and not fitted."]
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


def recount_unfitted(lyrics: LyricsVersion) -> int:
    """The version's count is always what its lines say now."""
    lyrics.unfitted = sum(1 for line in lyrics.lines if line.unfitted)
    return lyrics.unfitted


def fit_lines(lines: Sequence[str], melody: MelodyVersion, language: str,
              version: int = 1,
              previous: Optional[LyricsVersion] = None,
              sources: Optional[Sequence[str]] = None,
              required: Optional[Sequence[int]] = None) -> LyricsVersion:
    """Fit a list of written lines onto the melody's phrase slots.

    ``sources`` names who wrote each line ("creator", "lexicon",
    "llm:<name>"), by position; a line carried over from ``previous``
    keeps the source it had.  ``required`` lists the slot indices words
    were asked for: a blank or unsingable line there is unfitted, while
    a slot that was never asked for may stay empty without complaint.
    Without ``required``, every slot with text is expected to fit.
    """
    slots = build_slots(melody)
    lv = LyricsVersion(version=version, language=language,
                       melody_version=melody.version)
    locked_by_slot: Dict[int, LyricLine] = {}
    if previous:
        for i, line in enumerate(previous.lines):
            if line.locked and i < len(slots):
                locked_by_slot[i] = line
    asked = set(required) if required is not None else set()

    warnings: List[str] = []
    for i, slot in enumerate(slots):
        if i in locked_by_slot:
            lv.lines.append(locked_by_slot[i])
            continue
        text = lines[i] if i < len(lines) else ""
        syllables, indices, warn = fit_line(text, slot)
        warnings.extend(f"{slot.section_name}: {w}" for w in warn)
        unfitted = not syllables and (bool(text.strip()) or i in asked)
        if unfitted and not text.strip():
            warnings.append(f"{slot.section_name}: words were asked for and none "
                            f"came; the line is empty and unfitted.")
        source = ""
        if sources is not None and i < len(sources) and sources[i]:
            source = sources[i]
        elif previous is not None and i < len(previous.lines) \
                and previous.lines[i].text == text:
            source = previous.lines[i].source
        lv.lines.append(LyricLine(
            section_id=slot.section_id, text=text, syllables=syllables,
            note_indices=indices, start=slot.start, end=slot.end,
            source=source, unfitted=unfitted))
    lv.notes = "\n".join(warnings)
    recount_unfitted(lv)
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
    # Words were asked for this line by the act of writing them; a blank
    # typed on purpose is the creator emptying the line, not a failure.
    line.unfitted = not syllables and bool(new_text.strip())
    recount_unfitted(lyrics)
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
