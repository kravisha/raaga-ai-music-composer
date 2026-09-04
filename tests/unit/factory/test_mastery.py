"""Evidence rules: every kind, the three-failures drop, the L3 cap without
an application pass, and the next_test_level ladder."""
from __future__ import annotations

import pytest

from raagacomposer.factory.mastery import (apply_evidence, level_for_pass,
                                            next_test_level)
from raagacomposer.factory.models import (MasteryLevel, MasteryRecord, Split,
                                          TestLevel, TestResult)

pytestmark = pytest.mark.unit


def _record() -> MasteryRecord:
    return MasteryRecord(agent_id="agent_1", concept="add_s")


def _result(*, level: TestLevel, split: Split = Split.TRAINING,
           passed: bool = True) -> TestResult:
    return TestResult(test_id="test_1", agent_id="agent_1", lesson_id="lesson_1",
                      level=level, split=split, passed=passed,
                      score=1.0 if passed else 0.0)


# -- every evidence kind ----------------------------------------------------
def test_exposed_gives_l1():
    record = apply_evidence(_record(), "exposed", "lesson_1", True)
    assert record.level == MasteryLevel.L1_EXPOSED


def test_restated_gives_l2():
    record = apply_evidence(_record(), "restated", "reiter_1", True)
    assert record.level == MasteryLevel.L2_CAN_RESTATE


def test_explained_gives_l3():
    record = apply_evidence(_record(), "explained", "reiter_1", True)
    assert record.level == MasteryLevel.L3_CAN_EXPLAIN


def test_t3_gives_l4_with_result_evidence():
    record = apply_evidence(_record(), "T3", "result_1", True)
    assert record.level == MasteryLevel.L4_APPLY_WITH_GUIDANCE


def test_t4_validation_gives_l5():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "T4_validation", "result_2", True)
    assert record.level == MasteryLevel.L5_APPLY_INDEPENDENTLY


def test_t6_and_t7_give_l6():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "T6", "result_2", True)
    assert record.level == MasteryLevel.L6_DETECTS_ERRORS

    other = apply_evidence(_record(), "T3", "result_1", True)
    other = apply_evidence(other, "T7", "result_2", True)
    assert other.level == MasteryLevel.L6_DETECTS_ERRORS


def test_t8_and_t9_give_l7():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "T8", "result_2", True)
    assert record.level == MasteryLevel.L7_GENERALIZES

    other = apply_evidence(_record(), "T3", "result_1", True)
    other = apply_evidence(other, "T9", "result_2", True)
    assert other.level == MasteryLevel.L7_GENERALIZES


def test_authored_test_used_gives_l8():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "authored_test_used", "result_2", True)
    assert record.level == MasteryLevel.L8_CAN_TEACH


def test_field_gives_l9():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "field", "field_event_1", True)
    assert record.level == MasteryLevel.L9_EXPERT


# -- three failures drop one level -------------------------------------------
def test_three_failures_drop_one_level():
    record = apply_evidence(_record(), "T3", "result_1", True)
    assert record.level == MasteryLevel.L4_APPLY_WITH_GUIDANCE

    for i in range(2):
        record = apply_evidence(record, "T4_validation", f"result_fail_{i}", False)
        assert record.level == MasteryLevel.L4_APPLY_WITH_GUIDANCE  # not yet

    record = apply_evidence(record, "T4_validation", "result_fail_2", False)
    assert record.level == MasteryLevel.L3_CAN_EXPLAIN
    assert record.failures_at_level == 0


def test_failures_never_drop_below_l0():
    record = _record()
    for i in range(3):
        record = apply_evidence(record, "T3", f"result_fail_{i}", False)
    assert record.level == MasteryLevel.L0_UNKNOWN


def test_pass_resets_failure_streak():
    record = apply_evidence(_record(), "T3", "result_1", True)
    record = apply_evidence(record, "T4_validation", "result_fail_0", False)
    assert record.failures_at_level == 1
    record = apply_evidence(record, "T4_validation", "result_2", True)
    assert record.failures_at_level == 0
    assert record.level == MasteryLevel.L5_APPLY_INDEPENDENTLY


# -- the L3 cap: no application pass, no real mastery (TEST 7) --------------
def test_l3_cap_without_application_pass():
    record = _record()
    record = apply_evidence(record, "exposed", "lesson_1", True)
    record = apply_evidence(record, "restated", "reiter_1", True)
    record = apply_evidence(record, "explained", "reiter_1", True)
    assert record.level == MasteryLevel.L3_CAN_EXPLAIN

    # A kind that would normally grant L5, but with a non-result evidence
    # id: capped at L3, because nothing here was an actual graded test.
    record = apply_evidence(record, "T4_validation", "reiter_1", True)
    assert record.level == MasteryLevel.L3_CAN_EXPLAIN


def test_l3_cap_lifts_once_a_real_result_exists():
    record = _record()
    record = apply_evidence(record, "T3", "result_1", True)
    assert record.level == MasteryLevel.L4_APPLY_WITH_GUIDANCE
    record = apply_evidence(record, "T4_validation", "result_2", True)
    assert record.level == MasteryLevel.L5_APPLY_INDEPENDENTLY


# -- level_for_pass ----------------------------------------------------------
def test_level_for_pass_none_when_failed():
    assert level_for_pass(_result(level=TestLevel.T3_CONTROLLED_APPLICATION,
                                  passed=False)) is None


def test_level_for_pass_t4_needs_validation_or_hidden_split():
    training = _result(level=TestLevel.T4_INDEPENDENT_APPLICATION,
                       split=Split.TRAINING)
    assert level_for_pass(training) is None

    validation = _result(level=TestLevel.T4_INDEPENDENT_APPLICATION,
                         split=Split.VALIDATION)
    assert level_for_pass(validation) == MasteryLevel.L5_APPLY_INDEPENDENTLY

    hidden = _result(level=TestLevel.T4_INDEPENDENT_APPLICATION,
                     split=Split.HIDDEN)
    assert level_for_pass(hidden) == MasteryLevel.L5_APPLY_INDEPENDENTLY


def test_level_for_pass_t5_gives_nothing():
    assert level_for_pass(_result(level=TestLevel.T5_VARIATION)) is None


# -- next_test_level ladder ---------------------------------------------------
@pytest.mark.parametrize("level,expected", [
    (MasteryLevel.L0_UNKNOWN, TestLevel.T0_RECOGNITION),
    (MasteryLevel.L1_EXPOSED, TestLevel.T0_RECOGNITION),
    (MasteryLevel.L2_CAN_RESTATE, TestLevel.T1_RECALL),
    (MasteryLevel.L3_CAN_EXPLAIN, TestLevel.T3_CONTROLLED_APPLICATION),
    (MasteryLevel.L4_APPLY_WITH_GUIDANCE, TestLevel.T4_INDEPENDENT_APPLICATION),
    (MasteryLevel.L5_APPLY_INDEPENDENTLY, TestLevel.T5_VARIATION),
    (MasteryLevel.L6_DETECTS_ERRORS, TestLevel.T8_GENERALIZATION),
    (MasteryLevel.L7_GENERALIZES, TestLevel.T9_ADVERSARIAL),
    (MasteryLevel.L8_CAN_TEACH, TestLevel.T10_REAL_WORLD),
    (MasteryLevel.L9_EXPERT, TestLevel.T10_REAL_WORLD),
])
def test_next_test_level_ladder(level, expected):
    record = MasteryRecord(agent_id="agent_1", concept="add_s", level=level)
    assert next_test_level(record) == expected
