"""The temporary Judge (document 02 section 1C, document 06's ephemerality
requirement - TEST 5).

``convene`` is a function, not a class with a lifecycle: it builds a small
local object holding both sides' evidence, the shared knowledge and the
names of the rules it consulted, decides, and returns a ``Ruling``.  Nothing
here is kept after the call returns, and nothing here touches the store -
persistence is the caller's job (``cycle.py``), exactly as document 02
section 6 asks: persist the ruling, not the judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .models import Dispute, Ruling, TestResult
from .protocols import Rule


class UnresolvedDispute(Exception):
    """Raised by callers that require a resolved ruling and got none; the
    core itself never raises this - an unresolved ruling is a normal,
    valid outcome (document 02 section 5)."""


@dataclass
class _Judge:
    """The Judge's whole existence: one call's worth of context.  Built,
    consulted, discarded - there is no instance left to persist or hold a
    reference to after ``convene`` returns."""
    dispute: Dispute
    evidence_student: List[str] = field(default_factory=list)
    evidence_trainer: List[str] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)
    rule_names: List[str] = field(default_factory=list)


def convene(dispute: Dispute, rules: Sequence[Rule],
           knowledge: Sequence[str] = (),
           escalate: Optional[Callable[[Dispute], Optional[Ruling]]] = None
           ) -> Ruling:
    judge = _Judge(
        dispute=dispute,
        evidence_student=list(dispute.evidence_student),
        evidence_trainer=list(dispute.evidence_trainer),
        knowledge=list(dispute.shared_knowledge) + list(knowledge),
        rule_names=[r.name for r in rules])

    ruling: Optional[Ruling] = None
    decided_by = ""
    for rule in rules:
        if not rule.applies(judge.dispute):
            continue
        candidate = rule.decide(judge.dispute)
        if candidate is not None:
            ruling = candidate
            decided_by = rule.name
            break

    if ruling is None and escalate is not None:
        candidate = escalate(judge.dispute)
        if candidate is not None:
            ruling = candidate
            decided_by = "escalation"

    if ruling is None:
        ruling = Ruling(
            ruling="unresolved",
            needs_external_evidence=True,
            unresolved_issues=[judge.dispute.question] if judge.dispute.question
            else ["no rule or escalation could settle this dispute"],
            decided_by="")
    else:
        ruling.decided_by = decided_by

    ruling.dispute_id = judge.dispute.id
    return ruling
    # judge falls out of scope here - nothing keeps it alive.


def should_convene(result: TestResult, hard_rule_settles: bool,
                   min_confidence: float = 0.5, max_gap: float = 0.25) -> bool:
    """Document 02 section 4: invoke only when both sides are plausible and
    nothing settles it already.

    A result the trainer passed is not a dispute, whatever the two claims
    look like as text: the trainer accepted the work.  A dispute is the
    student standing by work the trainer rejected."""
    if hard_rule_settles:
        return False
    if result.passed:
        return False
    if result.student_claim == result.trainer_claim:
        return False
    if result.student_confidence < min_confidence:
        return False
    if result.trainer_confidence < min_confidence:
        return False
    gap = abs(result.student_confidence - result.trainer_confidence)
    return gap <= max_gap
