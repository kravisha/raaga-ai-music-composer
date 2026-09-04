"""Evidence rules for mastery (document 01 section 5, document 03 section 7's
"no false mastery" requirement - TEST 7 of the handoff).

A concept's mastery level only moves because of stated evidence: exposure,
an accepted reiteration, or a passed test at the right level and split.
Nothing here reads or writes the store; ``cycle.py`` supplies the evidence
after each step and persists the resulting record.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from .models import MasteryLevel, MasteryRecord, Split, TestLevel, TestResult

# Document 01 section 5's ladder, reached through document 03's test ladder
# and document 04's reiteration protocol.  "field" is real-world evidence
# (document 05 section 6), applied directly by the factory, not through a
# TestResult.
EVIDENCE_LEVELS: Dict[str, MasteryLevel] = {
    "exposed": MasteryLevel.L1_EXPOSED,
    "restated": MasteryLevel.L2_CAN_RESTATE,
    "explained": MasteryLevel.L3_CAN_EXPLAIN,
    "T3": MasteryLevel.L4_APPLY_WITH_GUIDANCE,
    "T4_validation": MasteryLevel.L5_APPLY_INDEPENDENTLY,
    "T6": MasteryLevel.L6_DETECTS_ERRORS,
    "T7": MasteryLevel.L6_DETECTS_ERRORS,
    "T8": MasteryLevel.L7_GENERALIZES,
    "T9": MasteryLevel.L7_GENERALIZES,
    "authored_test_used": MasteryLevel.L8_CAN_TEACH,
    "field": MasteryLevel.L9_EXPERT,
}


def kind_for_test(level: TestLevel, split: Split) -> Optional[str]:
    """The evidence kind a pass at this test level (and split) counts as, or
    None when a pass here does not by itself move mastery - T5 (variation)
    is diagnostic, and an independent-application pass only counts on a
    split the student was not trained on."""
    if level == TestLevel.T3_CONTROLLED_APPLICATION:
        return "T3"
    if level == TestLevel.T4_INDEPENDENT_APPLICATION:
        return "T4_validation" if split in (Split.VALIDATION, Split.HIDDEN) else None
    if level == TestLevel.T6_ERROR_DETECTION:
        return "T6"
    if level == TestLevel.T7_CORRECTION:
        return "T7"
    if level == TestLevel.T8_GENERALIZATION:
        return "T8"
    if level == TestLevel.T9_ADVERSARIAL:
        return "T9"
    return None


def level_for_pass(result: TestResult) -> Optional[MasteryLevel]:
    """The mastery level a passed result is evidence of, or None when this
    particular level/split combination is not itself evidence of a higher
    level (document 03 section 3: T4 counts for L5 only on a split the
    student was not trained on)."""
    if not result.passed:
        return None
    kind = kind_for_test(result.level, result.split)
    return EVIDENCE_LEVELS.get(kind) if kind else None


def apply_evidence(record: MasteryRecord, kind: str, evidence_id: str,
                   passed: bool = True) -> MasteryRecord:
    """Update ``record`` in place with one piece of evidence and return it.

    A pass raises the level to at least the evidence kind's level and clears
    the failure streak.  Three failures in a row at the current level drop
    it by one (never below L0).  TEST 7's cap: without at least one piece of
    evidence that actually came from a graded TestResult (its id starts with
    "result_"), the record never reports higher than L3 - restating and
    explaining a lesson is not the same as having applied it.
    """
    if passed:
        record.evidence = list(record.evidence) + [evidence_id]
        record.failures_at_level = 0
        target = EVIDENCE_LEVELS.get(kind)
        if target is not None:
            new_level = MasteryLevel(max(int(record.level), int(target)))
            if new_level > MasteryLevel.L3_CAN_EXPLAIN:
                has_test_evidence = any(
                    e.startswith("result_") for e in record.evidence)
                if not has_test_evidence:
                    new_level = MasteryLevel.L3_CAN_EXPLAIN
            record.level = new_level
    else:
        record.failures_at_level += 1
        if record.failures_at_level >= 3:
            record.level = MasteryLevel(
                max(int(MasteryLevel.L0_UNKNOWN), int(record.level) - 1))
            record.failures_at_level = 0
    record.updated_at = time.time()
    return record


def next_test_level(record: MasteryRecord) -> TestLevel:
    """The test level this record has earned the right to attempt next
    (document 03 section 2's ladder, gated by document 01 section 5's
    mastery).  ``AdaptiveTrainer`` reaches one rung higher still - T6 after
    T5, and generally the next rung after a calibrated streak - through
    ``should_raise_difficulty``; this function gives the entry point for the
    student's current level, not the ceiling."""
    level = record.level
    if level <= MasteryLevel.L1_EXPOSED:
        return TestLevel.T0_RECOGNITION
    if level == MasteryLevel.L2_CAN_RESTATE:
        return TestLevel.T1_RECALL
    if level == MasteryLevel.L3_CAN_EXPLAIN:
        return TestLevel.T3_CONTROLLED_APPLICATION
    if level == MasteryLevel.L4_APPLY_WITH_GUIDANCE:
        return TestLevel.T4_INDEPENDENT_APPLICATION
    if level == MasteryLevel.L5_APPLY_INDEPENDENTLY:
        return TestLevel.T5_VARIATION
    if level == MasteryLevel.L6_DETECTS_ERRORS:
        return TestLevel.T8_GENERALIZATION
    if level == MasteryLevel.L7_GENERALIZES:
        return TestLevel.T9_ADVERSARIAL
    return TestLevel.T10_REAL_WORLD
