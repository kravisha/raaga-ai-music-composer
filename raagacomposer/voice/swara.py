"""Prepare explicitly selected passages for sung swaras.

This is an opt-in rendering plan, not a new melody or a learning record.
The caller chooses the sections appropriate to the brief and owns saving,
undo, and acceptance of the returned draft. Instrumental sections are sung
only when explicitly selected; other instrumental sections remain silent.
"""
from __future__ import annotations

import re
from math import isfinite
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..core.models import ApprovalState, LyricLine, LyricsVersion, MelodyVersion
from ..core.versioning import LockedContentError
from ..lyrics.fitting import build_slots
from ..raaga.library import Raaga
from .renderer import SungSegment, plan_segments


_TOKEN = re.compile(r"(S|R[123]|G[123]|M[12]|P|D[123]|N[123])(?:\+*|-*)")
_SYLLABLES = {"S": "sa", "R": "ri", "G": "ga", "M": "ma",
              "P": "pa", "D": "dha", "N": "ni"}


def swara_syllable(token: str) -> str:
    """Sing the note name, not its variant number or octave marker.

    Variant and octave remain in the melody's pitch. R2 and G1 can share
    a pitch but keep their distinct sung names; no pitch-class guessing.
    """
    match = _TOKEN.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"Unrecognised swara token: {token!r}")
    return _SYLLABLES[match.group(1)[0]]


@dataclass
class SwaraSingingPlan:
    lyrics: LyricsVersion
    segments: List[SungSegment]
    section_ids: Tuple[str, ...]


def prepare_swara_singing(melody: MelodyVersion, raaga: Raaga, *,
                          section_ids: Sequence[str],
                          lyrics: Optional[LyricsVersion] = None
                          ) -> SwaraSingingPlan:
    """Make a new lyric draft and renderable segments for selected sections.

    Notes, timing, ornaments, original lyrics, and unrelated sections are
    unchanged. No section is selected implicitly. An empty selection keeps
    ordinary singing, without creating a new lyric version.

    Pass the resulting segments to ``voice.renderer.render``. Unlike its
    default melody wrapper, this plan can include explicitly requested
    swaras in an interlude without adding vocals to every prelude/interlude.
    Keep ``section_ids`` with the caller's vocal-direction state so subsequent
    renders can rebuild the same plan. This helper performs no persistence.
    """
    if lyrics is not None and lyrics.melody_version != melody.version:
        raise ValueError("These lyrics belong to a different melody version.")

    selected = set(section_ids)
    sections = {s.id: s for s in melody.sections}
    unknown = selected - sections.keys()
    if unknown:
        raise ValueError("A selected section no longer exists in this melody.")
    ordered_ids = tuple(s.id for s in melody.sections if s.id in selected)

    if not selected:
        draft = (deepcopy(lyrics) if lyrics is not None else
                 LyricsVersion(melody_version=melody.version))
        return SwaraSingingPlan(draft, plan_segments(melody, draft), ())

    if melody.raaga.casefold() not in {
            name.casefold() for name in (raaga.name, *raaga.aliases)}:
        raise ValueError("The supplied raaga does not match this melody.")
    if lyrics is not None and lyrics.state == ApprovalState.LOCKED:
        raise LockedContentError("The lyrics are locked. Unlock before adding swaras.")
    for sid in ordered_ids:
        if sections[sid].locked:
            raise LockedContentError(f"Section '{sections[sid].name}' is locked.")

    selected_indices = {i for i, note in enumerate(melody.notes)
                        if note.section_id in selected}
    for sid in ordered_ids:
        if not any(melody.notes[i].section_id == sid for i in selected_indices):
            raise ValueError(f"Section '{sections[sid].name}' has no notes to sing.")

    syllables = {}
    for i in sorted(selected_indices):
        note = melody.notes[i]
        syllables[i] = swara_syllable(note.swara)
        if not raaga.is_allowed(note.swara):
            raise ValueError(f"{note.swara} is not a permitted swara in {raaga.name}.")
        if note.midi != raaga.midi(note.swara, melody.tonic_midi):
            raise ValueError(f"The pitch and swara label disagree at note {i + 1}.")
        section = sections[note.section_id]
        if (not all(isfinite(t) for t in
                    (note.start, note.duration, section.start, section.end))
                or section.start < 0 or note.duration <= 0
                or note.start < section.start - 1e-6
                or note.end > section.end + 1e-6):
            raise ValueError(f"A note falls outside section '{section.name}'.")

    kept_lines = []
    for line in lyrics.lines if lyrics is not None else []:
        touches_selection = bool(selected_indices.intersection(line.note_indices))
        if line.section_id in selected:
            if line.locked:
                raise LockedContentError("A lyric line in the selected passage is locked.")
            if set(line.note_indices) - selected_indices:
                raise ValueError("A lyric line crosses the selected section boundary.")
        else:
            if touches_selection:
                raise ValueError("A lyric line crosses the selected section boundary.")
            kept_lines.append(deepcopy(line))

    for slot in build_slots(melody, include_instrumental=True):
        if slot.section_id not in selected:
            continue
        sung = [syllables[i] for i in slot.note_indices]
        kept_lines.append(LyricLine(
            section_id=slot.section_id, text=" ".join(sung), syllables=sung,
            note_indices=list(slot.note_indices), start=slot.start, end=slot.end))
    kept_lines.sort(key=lambda line: (line.start, line.end))
    names = ", ".join(sections[sid].name for sid in ordered_ids)
    notes = [lyrics.notes] if lyrics is not None and lyrics.notes else []
    notes.append(f"Sung swaras in: {names}.")
    draft = LyricsVersion(
        version=lyrics.version + 1 if lyrics is not None else 1,
        language=lyrics.language if lyrics is not None else "Tamil",
        melody_version=melody.version, lines=kept_lines,
        state=ApprovalState.DRAFT, notes="\n".join(notes))

    # The underlying planner yields one segment per note when its global
    # instrumental filter is off. Apply the more precise per-section choice.
    all_segments = plan_segments(melody, draft, vocal_sections_only=False)
    segments = []
    previous_end = -1.0
    for note, segment in zip(melody.notes, all_segments):
        section = sections.get(note.section_id)
        if (section is None or not section.kind.instrumental
                or note.section_id in selected):
            # Excluded accompaniment is not a preceding sung note.
            segment.legato = (segment.start - previous_end) < 0.06
            segments.append(segment)
            previous_end = segment.end
    return SwaraSingingPlan(draft, segments, ordered_ids)
