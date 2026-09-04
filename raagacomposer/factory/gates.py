"""Document 03 section 7's promotion gate and document 05 section 5's release
gate.  Both read the store; neither writes to it - a gate is a question, not
an event."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import (AgentProfile, DisputeStatus, GateReport, MasteryLevel,
                     Split)
from .store import FactoryStore


@dataclass
class GateThresholds:
    min_mastery: MasteryLevel = MasteryLevel.L5_APPLY_INDEPENDENTLY
    require_hidden_pass: bool = True
    stable_repeats: int = 3
    calibration_tolerance: float = 0.2


def _critical_unresolved(store: FactoryStore, profile: AgentProfile,
                         capability: str = "") -> List:
    out = []
    for d in store.disputes(agent_id=profile.id):
        if not d.critical or d.status == DisputeStatus.RESOLVED:
            continue
        if capability and d.lesson_id:
            lesson = store.lesson(d.lesson_id)
            if lesson is not None and lesson.concept != capability:
                continue
        out.append(d)
    return out


def promotion_gate(store: FactoryStore, profile: AgentProfile, capability: str,
                   thresholds: GateThresholds = GateThresholds()) -> GateReport:
    checks = {}
    reasons: List[str] = []

    record = store.mastery(profile.id, capability)
    checks["mastery"] = int(record.level) >= int(thresholds.min_mastery)
    if not checks["mastery"]:
        reasons.append(f"{capability}: mastery is {record.level.label}, "
                       f"needs at least {thresholds.min_mastery.label}")

    critical = _critical_unresolved(store, profile, capability)
    checks["no_critical_unresolved"] = not critical
    if critical:
        reasons.append(f"{capability}: {len(critical)} critical dispute(s) "
                       f"unresolved")

    results = store.results(profile.id, capability=capability, limit=1000)
    checks["unseen_success"] = any(
        r.passed and r.split in (Split.HIDDEN, Split.VALIDATION)
        for r in results)
    if not checks["unseen_success"]:
        reasons.append(f"{capability}: no passed result on a hidden or "
                       f"validation test")

    recent = results[:thresholds.stable_repeats]
    checks["stable"] = (len(recent) >= thresholds.stable_repeats
                        and all(r.passed for r in recent))
    if not checks["stable"]:
        reasons.append(f"{capability}: last {thresholds.stable_repeats} "
                       f"results are not all passing")

    checks["calibrated"] = (len(recent) >= thresholds.stable_repeats and all(
        abs(r.student_confidence - r.score) <= thresholds.calibration_tolerance
        for r in recent))
    if not checks["calibrated"]:
        reasons.append(f"{capability}: confidence is not calibrated to "
                       f"correctness")

    return GateReport(gate="promotion", passed=all(checks.values()),
                      checks=checks, reasons=reasons)


def release_gate(store: FactoryStore, profile: AgentProfile,
                 thresholds: GateThresholds = GateThresholds()) -> GateReport:
    checks = {}
    reasons: List[str] = []

    if not profile.capabilities:
        checks["capabilities_pass"] = False
        reasons.append("profile states no capabilities to release")
    else:
        all_pass = True
        for capability in profile.capabilities:
            report = promotion_gate(store, profile, capability, thresholds)
            if not report.passed:
                all_pass = False
                reasons.extend(report.reasons)
            if thresholds.require_hidden_pass:
                hidden = store.results(profile.id, capability=capability,
                                       split=Split.HIDDEN, limit=1000)
                if not any(r.passed for r in hidden):
                    all_pass = False
                    reasons.append(f"{capability}: no passed hidden test")
        checks["capabilities_pass"] = all_pass

    critical = _critical_unresolved(store, profile)
    checks["no_critical_unresolved_dispute"] = not critical
    if critical:
        reasons.append(f"{len(critical)} critical dispute(s) unresolved")

    spec = profile.spec
    checks["rollback_stated"] = bool(spec and spec.rollback)
    if not checks["rollback_stated"]:
        reasons.append("no rollback plan stated")

    checks["permissions_bounded"] = bool(spec and spec.permissions)
    if not checks["permissions_bounded"]:
        reasons.append("no permissions stated")

    checks["monitoring_stated"] = bool(spec and spec.monitoring)
    if not checks["monitoring_stated"]:
        reasons.append("no monitoring stated")

    checks["escalation_stated"] = bool(spec and spec.escalation)
    if not checks["escalation_stated"]:
        reasons.append("no escalation behaviour stated")

    checks["knowledge_version_set"] = bool(profile.knowledge_version)
    if not checks["knowledge_version_set"]:
        reasons.append("knowledge_version is not set")

    return GateReport(gate="release", passed=all(checks.values()),
                      checks=checks, reasons=reasons)
