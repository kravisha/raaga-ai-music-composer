"""Originality safeguards (learning specification section 12).

The point of listening is to learn grammar, not to reproduce melodies.  Every
learned phrase is indexed by overlapping n-grams of its swara sequence; any
generated line is checked against that index before it is offered, and a run
that is too long and too identical is rejected and regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from .knowledge import KnowledgeRepository, Phrase

log = get_logger("agent.originality")

DEFAULT_NGRAM = 5
DEFAULT_MAX_RUN = 6


@dataclass
class OriginalityReport:
    longest_match: int = 0
    matched_phrase_id: str = ""
    matched_source_id: str = ""
    matched_swaras: List[str] = field(default_factory=list)
    total_notes: int = 0
    is_original: bool = True
    score: float = 1.0
    reason: str = ""

    def summary(self) -> str:
        if self.is_original:
            return (f"original (longest shared run {self.longest_match} of "
                    f"{self.total_notes})")
        return (f"too close to a learned phrase: {self.longest_match} notes in a "
                f"row match {' '.join(self.matched_swaras)}")


def _base(token: str) -> str:
    """Octave-insensitive swara, so transposed copying is still caught."""
    return token.replace("+", "").replace("-", "")


class PhraseIndex:
    """N-gram index over learned phrases."""

    def __init__(self, n: int = DEFAULT_NGRAM) -> None:
        self.n = max(2, int(n))
        self._grams: Dict[Tuple[str, ...], List[str]] = {}
        self._phrases: Dict[str, Phrase] = {}

    def add(self, phrase: Phrase) -> None:
        swaras = [_base(s) for s in phrase.swaras]
        if len(swaras) < self.n:
            return
        self._phrases[phrase.id] = phrase
        for i in range(len(swaras) - self.n + 1):
            gram = tuple(swaras[i:i + self.n])
            self._grams.setdefault(gram, []).append(phrase.id)

    def add_many(self, phrases: Iterable[Phrase]) -> None:
        for phrase in phrases:
            self.add(phrase)

    @classmethod
    def from_repository(cls, repo: KnowledgeRepository, raaga: str = "",
                        n: int = DEFAULT_NGRAM) -> "PhraseIndex":
        index = cls(n)
        index.add_many(repo.phrases(raaga=raaga, limit=5000))
        return index

    @property
    def size(self) -> int:
        return len(self._phrases)

    def phrase(self, phrase_id: str) -> Optional[Phrase]:
        return self._phrases.get(phrase_id)

    # -- checking ----------------------------------------------------------
    def longest_shared_run(self, swaras: Sequence[str]
                           ) -> Tuple[int, str, List[str]]:
        """Longest run of notes shared with any indexed phrase."""
        seq = [_base(s) for s in swaras]
        if len(seq) < self.n or not self._grams:
            return 0, "", []

        best_len, best_id, best_run = 0, "", []
        for i in range(len(seq) - self.n + 1):
            gram = tuple(seq[i:i + self.n])
            for phrase_id in self._grams.get(gram, ()):
                phrase = self._phrases.get(phrase_id)
                if phrase is None:
                    continue
                target = [_base(s) for s in phrase.swaras]
                run = self._extend(seq, i, target, gram)
                if run > best_len:
                    best_len, best_id = run, phrase_id
                    best_run = seq[i:i + run]
        return best_len, best_id, best_run

    @staticmethod
    def _extend(seq: List[str], start: int, target: List[str],
                gram: Tuple[str, ...]) -> int:
        """How far the match at *start* extends inside *target*."""
        n = len(gram)
        best = 0
        for j in range(len(target) - n + 1):
            if tuple(target[j:j + n]) != gram:
                continue
            length = n
            while (start + length < len(seq) and j + length < len(target)
                   and seq[start + length] == target[j + length]):
                length += 1
            best = max(best, length)
        return best


def check(swaras: Sequence[str], index: PhraseIndex,
          max_run: int = DEFAULT_MAX_RUN,
          repo: Optional[KnowledgeRepository] = None) -> OriginalityReport:
    """Is this line the agent's own, or is it repeating something it heard?"""
    report = OriginalityReport(total_notes=len(swaras))
    if not swaras or index.size == 0:
        report.reason = "nothing learned to copy from yet"
        return report

    run, phrase_id, matched = index.longest_shared_run(swaras)
    report.longest_match = run
    report.matched_phrase_id = phrase_id
    report.matched_swaras = matched
    if phrase_id and repo is not None:
        phrase = index.phrase(phrase_id)
        if phrase is not None:
            report.matched_source_id = phrase.source_id

    ratio = run / max(1, len(swaras))
    report.score = round(max(0.0, 1.0 - max(0.0, run - 2) / max(3.0, max_run)), 3)
    report.is_original = run <= max_run and ratio < 0.6
    if not report.is_original:
        report.reason = (f"{run} notes in a row match a learned phrase "
                         f"({ratio:.0%} of the line)")
    return report


def most_similar(swaras: Sequence[str], index: PhraseIndex) -> Optional[Phrase]:
    _, phrase_id, _ = index.longest_shared_run(swaras)
    return index.phrase(phrase_id) if phrase_id else None
