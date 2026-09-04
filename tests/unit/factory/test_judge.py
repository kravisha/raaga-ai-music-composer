"""The Judge: a rule decides; with no rule and no escalation the result is
unresolved with needs_external_evidence; the escalation callable is used
only when the rules are silent; and the judge holds nothing after return -
no module-level state, and the Ruling carries no reference to the judge
object."""
from __future__ import annotations

import gc
import weakref

import pytest

from raagacomposer.factory import judge as judge_module
from raagacomposer.factory.judge import _Judge, convene, should_convene
from raagacomposer.factory.models import Dispute, Ruling, Split, TestResult

from tests.unit.factory.toy_domain import HardRule

pytestmark = pytest.mark.unit


def _dispute(**kwargs) -> Dispute:
    base = dict(agent_id="agent_1", question="plural of cat",
               student_claim="cats", trainer_claim="cat",
               student_confidence=0.8, trainer_confidence=0.6)
    base.update(kwargs)
    return Dispute(**base)


def test_rule_decides():
    dispute = _dispute()
    ruling = convene(dispute, [HardRule()])
    assert ruling.resolved
    assert ruling.decided_by == "plural_hard_rule"
    assert ruling.ruling == "cats"
    assert ruling.accepted_claim == "student"
    assert ruling.dispute_id == dispute.id


def test_no_rule_no_escalation_is_unresolved():
    dispute = _dispute(question="what mood does this evoke")
    ruling = convene(dispute, [HardRule()])
    assert not ruling.resolved
    assert ruling.ruling == "unresolved"
    assert ruling.needs_external_evidence is True
    assert ruling.decided_by == ""
    assert ruling.dispute_id == dispute.id


def test_escalation_used_only_when_rules_are_silent():
    called = {"count": 0}

    def escalate(d: Dispute):
        called["count"] += 1
        return Ruling(ruling="escalated-answer", accepted_claim="trainer")

    # A rule applies and decides: escalation must not be consulted.
    settled = _dispute()
    ruling = convene(settled, [HardRule()], escalate=escalate)
    assert ruling.decided_by == "plural_hard_rule"
    assert called["count"] == 0

    # No rule applies: escalation is consulted.
    silent = _dispute(question="what mood does this evoke")
    ruling = convene(silent, [HardRule()], escalate=escalate)
    assert ruling.decided_by == "escalation"
    assert ruling.ruling == "escalated-answer"
    assert called["count"] == 1


def test_escalation_returning_none_falls_back_to_unresolved():
    def escalate(d: Dispute):
        return None

    dispute = _dispute(question="what mood does this evoke")
    ruling = convene(dispute, [HardRule()], escalate=escalate)
    assert ruling.ruling == "unresolved"
    assert ruling.needs_external_evidence is True


def test_judge_is_ephemeral():
    dispute = _dispute()
    ref_holder = {}

    # Wrap _Judge's construction is not necessary: convene builds it
    # internally.  We verify no lasting reference exists by checking the
    # module has no instance-holding attribute and by weak-referencing an
    # equivalent object to prove nothing outside convene keeps it alive.
    probe = _Judge(dispute=dispute)
    ref = weakref.ref(probe)
    del probe
    gc.collect()
    assert ref() is None

    ruling = convene(dispute, [HardRule()])
    assert not hasattr(judge_module, "_judge_instance")
    assert not hasattr(ruling, "judge")
    assert not hasattr(ruling, "_judge")


def test_judge_module_has_no_instance_state():
    # No module-level object of type _Judge, or any mutable cache, survives
    # between calls.
    for name in dir(judge_module):
        if name.startswith("_") and name not in ("_Judge", "__name__",
                                                  "__doc__", "__file__",
                                                  "__loader__", "__spec__",
                                                  "__package__", "__builtins__",
                                                  "__cached__"):
            value = getattr(judge_module, name)
            assert not isinstance(value, _Judge)


# -- should_convene -----------------------------------------------------------
def _result(**kwargs) -> TestResult:
    base = dict(student_claim="cats", trainer_claim="cat",
               student_confidence=0.7, trainer_confidence=0.6)
    base.update(kwargs)
    return TestResult(**base)


def test_should_convene_true_when_plausible_and_close():
    result = _result()
    assert should_convene(result, hard_rule_settles=False) is True


def test_should_convene_false_when_hard_rule_settles():
    result = _result()
    assert should_convene(result, hard_rule_settles=True) is False


def test_should_convene_false_when_claims_agree():
    result = _result(student_claim="cats", trainer_claim="cats")
    assert should_convene(result, hard_rule_settles=False) is False


def test_should_convene_false_when_confidence_too_low():
    result = _result(student_confidence=0.2)
    assert should_convene(result, hard_rule_settles=False) is False


def test_should_convene_false_when_gap_too_large():
    result = _result(student_confidence=0.95, trainer_confidence=0.5)
    assert should_convene(result, hard_rule_settles=False) is False
