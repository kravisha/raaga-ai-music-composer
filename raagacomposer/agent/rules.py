"""Hard rules for the Judge (docs/PLAN_agent_factory.md, the Raga pilot).

Each rule decides a dispute from the raaga library alone - never from what
either side claims - and returns ``None`` when the question is outside its
competence (a mood claim, a phrase's authenticity), so it is left to
escalation or stays unresolved.  A learned fact that contradicts the library
is heuristic knowledge; these rules only ever consult the library, so a
learned claim that disagrees with it simply does not match the verdict and
loses (``FactRule``).

Evidence convention: a dispute that concerns a phrase carries evidence lines
"phrase: S R2 G2 ..." and "raaga: <name>" (``RagaStudent.perform`` and
``RagaTrainer.grade`` both write lines in this shape); these rules read them
from either side's evidence list.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..factory.models import Dispute, KnowledgeClass, Ruling
from ..raaga.library import SWARA_SEMITONES, Raaga, RaagaLibrary, parse_swara
from .originality import PhraseIndex, check as check_originality

TONIC = 60

FACT_QUESTION = re.compile(r"what is the (\w+) of (.+?)\??$", re.I)


def _evidence_value(evidence: Sequence[str], prefix: str) -> str:
    for line in list(evidence):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _phrase_and_raaga(dispute: Dispute, library: RaagaLibrary
                      ) -> Optional[Tuple[List[str], Raaga]]:
    # The trainer's evidence first: when it claims a violation it names the
    # phrase it objects to, and that is the phrase the ruling is about.
    lines = list(dispute.evidence_trainer) + list(dispute.evidence_student) \
        + list(dispute.shared_knowledge)
    phrase_text = _evidence_value(lines, "phrase:")
    raaga_name = _evidence_value(lines, "raaga:")
    if not phrase_text or not raaga_name:
        return None
    raaga = library.get(raaga_name)
    if raaga is None:
        return None
    tokens = phrase_text.split()
    # A line that is not made of swaras is not a phrase; a rule that
    # judged a stray word would be ruling on nothing.
    if not tokens or any(parse_swara(t)[0] not in SWARA_SEMITONES
                         for t in tokens):
        return None
    return tokens, raaga


def _normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _base_tokens(text: str) -> Set[str]:
    return {parse_swara(t)[0] for t in (text or "").split()}


def _reference_value(raaga: Raaga, key: str) -> str:
    return {
        "name": raaga.name,
        "arohanam": " ".join(raaga.arohanam),
        "avarohanam": " ".join(raaga.avarohanam),
        "swaras": " ".join(raaga.allowed),
        "jeeva": " ".join(raaga.jeeva),
        "nyasa": " ".join(raaga.nyasa),
        "graha": " ".join(raaga.graha),
    }.get(key, "")


def _direction_valid(raaga: Raaga, tokens: Sequence[str]) -> bool:
    """The engine's own walk (``PracticeEngine._judge_valid``'s logic):
    every move must be permitted by the arohanam going up and the
    avarohanam coming down."""
    ascending_ok = set(raaga.ascending)
    descending_ok = set(raaga.descending)
    bases = [parse_swara(t)[0] for t in tokens]
    midi = [raaga.midi(t, TONIC) for t in tokens]
    for i in range(1, len(tokens)):
        if midi[i] > midi[i - 1] and bases[i] not in ascending_ok:
            return False
        if midi[i] < midi[i - 1] and bases[i] not in descending_ok:
            return False
    return True


def _ruling_from_verdict(dispute: Dispute, verdict: str, rationale: str,
                         rule_name: str, confidence: float = 1.0,
                         correction: str = "") -> Ruling:
    """Build a ``Ruling`` that accepts whichever side's claim matches the
    hard verdict, or "neither" when both agree or both are wrong.

    ``correction`` is what the losing side is told; when it names a fact
    ("<raaga>'s <key> is <value> by the library (hard knowledge)") the
    student stores it and the contradicting claim is overruled.  It
    defaults to the rationale."""
    correction = correction or rationale
    student_ok = _normalise(dispute.student_claim) == _normalise(verdict)
    trainer_ok = _normalise(dispute.trainer_claim) == _normalise(verdict)
    if student_ok and not trainer_ok:
        accepted, rejected = "student", "trainer"
    elif trainer_ok and not student_ok:
        accepted, rejected = "trainer", "student"
    else:
        accepted, rejected = "neither", ""
    return Ruling(
        dispute_id=dispute.id, ruling=verdict, accepted_claim=accepted,
        rejected_claim=rejected, rationale=rationale, confidence=confidence,
        correction_student="" if student_ok else correction,
        correction_trainer="" if trainer_ok else correction,
        decided_by=rule_name)


class AllowedSwarasRule:
    """Is every swara in the phrase one the raaga's library entry allows?"""

    def __init__(self, library: RaagaLibrary) -> None:
        self.library = library

    @property
    def name(self) -> str:
        return "AllowedSwarasRule"

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def applies(self, dispute: Dispute) -> bool:
        low = (dispute.question or "").lower()
        return ("is the phrase" in low or "valid in" in low) and \
            _phrase_and_raaga(dispute, self.library) is not None

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        if not self.applies(dispute):
            return None
        found = _phrase_and_raaga(dispute, self.library)
        if found is None:
            return None
        tokens, raaga = found
        allowed = set(raaga.allowed)
        bases = [parse_swara(t)[0] for t in tokens]
        outside = sorted({b for b in bases if b not in allowed})
        verdict = "invalid" if outside else "valid"
        rationale = (f"{', '.join(outside)} not in {raaga.name}'s allowed "
                     f"swaras ({', '.join(raaga.allowed)})" if outside else
                     f"every swara in the phrase belongs to {raaga.name}")
        # The durable lesson is the swara inventory itself, so a student
        # that believed in a wrong note stores the right set.
        correction = (f"{raaga.name}'s swaras is {' '.join(raaga.allowed)} "
                      f"by the library (hard knowledge)")
        return _ruling_from_verdict(dispute, verdict, rationale, self.name,
                                    correction=correction)


class DirectionRule:
    """Does the phrase respect the arohanam going up and the avarohanam
    coming down?"""

    def __init__(self, library: RaagaLibrary) -> None:
        self.library = library

    @property
    def name(self) -> str:
        return "DirectionRule"

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def applies(self, dispute: Dispute) -> bool:
        low = (dispute.question or "").lower()
        return ("arohanam" in low or "avarohanam" in low or "direction" in low) \
            and _phrase_and_raaga(dispute, self.library) is not None

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        if not self.applies(dispute):
            return None
        found = _phrase_and_raaga(dispute, self.library)
        if found is None:
            return None
        tokens, raaga = found
        valid = _direction_valid(raaga, tokens)
        verdict = "valid" if valid else "invalid"
        rationale = (f"every move follows {raaga.name}'s arohanam/avarohanam"
                     if valid else
                     f"a move in this phrase is not permitted by "
                     f"{raaga.name}'s arohanam ({' '.join(raaga.arohanam)}) "
                     f"or avarohanam ({' '.join(raaga.avarohanam)})")
        return _ruling_from_verdict(dispute, verdict, rationale, self.name)


class RestingNoteRule:
    """Does the phrase come to rest on one of the raaga's nyasa swaras?"""

    def __init__(self, library: RaagaLibrary) -> None:
        self.library = library

    @property
    def name(self) -> str:
        return "RestingNoteRule"

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def applies(self, dispute: Dispute) -> bool:
        low = (dispute.question or "").lower()
        return ("rest on" in low or "cadence" in low) and \
            _phrase_and_raaga(dispute, self.library) is not None

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        if not self.applies(dispute):
            return None
        found = _phrase_and_raaga(dispute, self.library)
        if found is None:
            return None
        tokens, raaga = found
        last = parse_swara(tokens[-1])[0]
        nyasa = set(raaga.nyasa)
        verdict = "valid" if last in nyasa else "invalid"
        rationale = (f"{last} is a resting note (nyasa) of {raaga.name}: "
                     f"{', '.join(raaga.nyasa)}" if verdict == "valid" else
                     f"{last} is not one of {raaga.name}'s resting notes "
                     f"({', '.join(raaga.nyasa) or 'none defined'})")
        return _ruling_from_verdict(dispute, verdict, rationale, self.name)


class FactRule:
    """"What is the <key> of <raaga>?" - the library's value is hard; a
    learned value that differs from it loses."""

    def __init__(self, library: RaagaLibrary) -> None:
        self.library = library

    @property
    def name(self) -> str:
        return "FactRule"

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def _lookup(self, dispute: Dispute) -> Optional[Tuple[str, str, Raaga]]:
        match = FACT_QUESTION.search(dispute.question or "")
        if not match:
            return None
        key = match.group(1).strip().lower()
        raaga = self.library.get(match.group(2).strip())
        if raaga is None:
            return None
        value = _reference_value(raaga, key)
        if not value:
            return None
        return key, value, raaga

    def applies(self, dispute: Dispute) -> bool:
        return self._lookup(dispute) is not None

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        found = self._lookup(dispute)
        if found is None:
            return None
        key, value, raaga = found
        rationale = f"{raaga.name}'s {key} is {value} by the library (hard knowledge)"
        return _ruling_from_verdict(dispute, value, rationale, self.name)


class OriginalityRule:
    """Is the phrase the agent's own, or too close to a phrase it has
    heard?  Needs the raaga's ``PhraseIndex`` built at construction; a rule
    with no index built for it never applies (nothing to check against)."""

    def __init__(self, library: RaagaLibrary,
                phrase_index: Optional[PhraseIndex] = None) -> None:
        self.library = library
        self.phrase_index = phrase_index

    @property
    def name(self) -> str:
        return "OriginalityRule"

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def applies(self, dispute: Dispute) -> bool:
        if self.phrase_index is None:
            return False
        low = (dispute.question or "").lower()
        return "original" in low and \
            _phrase_and_raaga(dispute, self.library) is not None

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        if not self.applies(dispute):
            return None
        found = _phrase_and_raaga(dispute, self.library)
        if found is None:
            return None
        tokens, _raaga = found
        report = check_originality(tokens, self.phrase_index)
        verdict = "valid" if report.is_original else "invalid"
        return _ruling_from_verdict(dispute, verdict, report.summary(), self.name)


def hard_rules(library: RaagaLibrary,
               phrase_index: Optional[PhraseIndex] = None) -> List:
    """The Raga pilot's rule set for ``factory.judge.convene``."""
    return [
        AllowedSwarasRule(library),
        DirectionRule(library),
        RestingNoteRule(library),
        FactRule(library),
        OriginalityRule(library, phrase_index),
    ]
