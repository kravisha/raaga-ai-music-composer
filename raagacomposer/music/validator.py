"""Melody validator (spec sections 12.14, 20 phase D).

Checks a generated tune against the structural raaga rules before the creator is
asked to approve it.  Findings are advisory text plus a 0..1 fidelity score;
nothing is silently rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core.models import MelodyVersion, Note
from ..raaga.library import Raaga, parse_swara


@dataclass
class ValidationReport:
    score: float = 1.0
    issues: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        head = f"Raaga fidelity {self.score * 100:.0f}%"
        if not self.issues:
            return head + " - no issues found."
        return head + "\n" + "\n".join(f"- {i}" for i in self.issues)


def validate(melody: MelodyVersion, raaga: Raaga,
             voice_low: int = 48, voice_high: int = 84) -> ValidationReport:
    report = ValidationReport()
    notes = sorted(melody.notes, key=lambda n: n.start)
    if not notes:
        report.score = 0.0
        report.issues.append("The tune has no notes.")
        return report

    allowed = set(raaga.allowed)
    ascending_ok = set(raaga.ascending)
    descending_ok = set(raaga.descending)
    forbidden = set(raaga.forbidden_swaras)

    out_of_raaga: List[str] = []
    aro_breaks = 0
    ava_breaks = 0
    out_of_range = 0

    for i, n in enumerate(notes):
        base, _ = parse_swara(n.swara)
        if base in forbidden or base not in allowed:
            out_of_raaga.append(f"{n.swara}@{n.start:.1f}s")
        if not (voice_low <= n.midi <= voice_high):
            out_of_range += 1
        if i:
            prev = notes[i - 1]
            if n.midi > prev.midi and base not in ascending_ok:
                aro_breaks += 1
            elif n.midi < prev.midi and base not in descending_ok:
                ava_breaks += 1

    penalties = 0.0
    if out_of_raaga:
        report.issues.append(
            f"{len(out_of_raaga)} note(s) outside {raaga.name}: "
            + ", ".join(out_of_raaga[:6]) + ("..." if len(out_of_raaga) > 6 else ""))
        penalties += min(0.5, 0.05 * len(out_of_raaga))
    if aro_breaks:
        report.issues.append(
            f"{aro_breaks} ascending move(s) use a note not in the arohanam.")
        penalties += min(0.25, 0.01 * aro_breaks)
    if ava_breaks:
        report.issues.append(
            f"{ava_breaks} descending move(s) use a note not in the avarohanam.")
        penalties += min(0.25, 0.01 * ava_breaks)
    if out_of_range:
        report.issues.append(
            f"{out_of_range} note(s) fall outside the selected voice range.")
        penalties += min(0.15, 0.01 * out_of_range)

    # Forbidden multi-note phrases.
    bases = [parse_swara(n.swara)[0] for n in notes]
    for phrase in raaga.avoid:
        if len(phrase) < 2:
            continue
        hits = _count_subsequence(bases, [parse_swara(p)[0] for p in phrase])
        if hits:
            report.issues.append(
                f"Avoided phrase {' '.join(phrase)} appears {hits} time(s).")
            penalties += min(0.2, 0.04 * hits)

    # Cadences: each section should settle on a resting note.
    if raaga.nyasa:
        bad = []
        for section in melody.sections:
            tail = [n for n in notes if n.section_id == section.id]
            if not tail:
                continue
            if parse_swara(tail[-1].swara)[0] not in raaga.nyasa:
                bad.append(section.name)
        if bad:
            report.issues.append(
                "Section(s) not resting on a nyasa swara: " + ", ".join(bad))
            penalties += min(0.15, 0.03 * len(bad))

    jeeva_hits = sum(1 for b in bases if b in set(raaga.jeeva))
    report.stats = {
        "notes": len(notes),
        "duration": round(melody.duration, 2),
        "jeeva_ratio": round(jeeva_hits / len(bases), 3),
        "range": f"{min(n.midi for n in notes)}-{max(n.midi for n in notes)}",
        "out_of_raaga": len(out_of_raaga),
    }
    if raaga.jeeva and jeeva_hits / len(bases) < 0.12:
        report.issues.append(
            f"The life-giving notes ({', '.join(raaga.jeeva)}) are underused; "
            "the raaga may not be recognisable.")
        penalties += 0.08

    report.score = max(0.0, 1.0 - penalties)
    return report


def _count_subsequence(haystack: List[str], needle: List[str]) -> int:
    if not needle or len(needle) > len(haystack):
        return 0
    return sum(1 for i in range(len(haystack) - len(needle) + 1)
               if haystack[i:i + len(needle)] == needle)


def repair(melody: MelodyVersion, raaga: Raaga) -> int:
    """Snap out-of-raaga notes to the nearest allowed pitch. Returns how many."""
    fixed = 0
    allowed = set(raaga.allowed)
    for n in melody.notes:
        base, _ = parse_swara(n.swara)
        if base not in allowed:
            token = raaga.nearest_token(n.midi, melody.tonic_midi)
            n.swara = token
            n.midi = raaga.midi(token, melody.tonic_midi)
            fixed += 1
    return fixed
