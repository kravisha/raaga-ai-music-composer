"""Document 05's lifecycle: bootstrap an agent, train it, assess its
maturity, turn field experience into a candidate lesson, and let the
factory notice what worked across every agent it has built."""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from .cycle import CycleOutcome, LearningCycle
from .mastery import apply_evidence
from .models import (AgentProfile, AgentSpec, KnowledgeClass, MasteryLevel,
                     Maturity, Promotion, ReusableLesson, Split,
                     ValidationStatus)
from .protocols import Rule, Student, Trainer
from .store import FactoryStore


class AgentFactory:
    """Document 05 sections 2 to 7."""

    def __init__(self, store: FactoryStore) -> None:
        self.store = store

    # -- bootstrap ---------------------------------------------------------
    def create(self, spec: AgentSpec, curriculum: str = "",
              knowledge_version: str = "") -> AgentProfile:
        """Document 05 section 2: a new agent begins with the validated
        reusable lessons already in scope for its domain, not from nothing."""
        profile = AgentProfile(
            name=spec.name, role=spec.role, domain=spec.domain,
            capabilities=list(spec.capabilities), current_curriculum=curriculum,
            knowledge_version=knowledge_version, spec=spec)

        preloaded = [
            r for r in self.store.reusable_lessons(domain=spec.domain)
            if r.validation_status in (ValidationStatus.VALIDATED,
                                       ValidationStatus.SHARED)]
        for reusable in preloaded:
            record = self.store.mastery(profile.id, reusable.scope_concept)
            record = apply_evidence(record, "exposed", reusable.id, True)
            self.store.save_mastery(record)
            label = f"preloaded:{reusable.scope_concept}"
            if label not in profile.strengths:
                profile.strengths.append(label)

        self.store.save_profile(profile)
        self.store.record_metric("bootstrap.preloaded", float(len(preloaded)),
                                 agent_id=profile.id, detail=spec.domain)
        return profile

    def preloaded_lessons(self, profile: AgentProfile) -> List[ReusableLesson]:
        markers = {s[len("preloaded:"):] for s in profile.strengths
                  if s.startswith("preloaded:")}
        return [r for r in self.store.reusable_lessons(domain=profile.domain)
               if r.scope_concept in markers]

    # -- training ------------------------------------------------------------
    def train(self, profile: AgentProfile, student: Student, trainer: Trainer,
             rules: Sequence[Rule] = (), escalate: Optional[Callable] = None,
             max_cycles: int = 10) -> List[CycleOutcome]:
        cycle = LearningCycle(self.store, student, trainer, rules=rules,
                              escalate=escalate)
        return cycle.run_until(max_cycles)

    # -- maturity (document 05 section 4) -------------------------------------
    def assess(self, profile: AgentProfile) -> Maturity:
        capabilities = profile.capabilities
        mastery_table = self.store.mastery_table(profile.id)

        def level(concept: str) -> MasteryLevel:
            record = mastery_table.get(concept)
            return record.level if record else MasteryLevel.L0_UNKNOWN

        results_by_capability = {
            c: self.store.results(profile.id, capability=c, limit=1000)
            for c in capabilities}

        def low_supervision() -> bool:
            last20 = self.store.results(profile.id, limit=20)
            if not last20:
                return True
            cutoff = min(r.at for r in last20)
            return not any(m["at"] >= cutoff
                           for m in self.store.metrics("remediation")
                           if m["agent_id"] == profile.id)

        # The pipeline is a ladder (document 05 section 4): a stage is
        # reached only when every stage below it holds, so three passed
        # recall tests cannot make an agent "reliable" that cannot yet
        # apply anything, and field trust presumes correctness.
        ladder = [
            (Maturity.S1_EFFECTIVE, lambda: any(
                level(c) >= MasteryLevel.L4_APPLY_WITH_GUIDANCE
                for c in capabilities)),
            (Maturity.S2_WORKING, lambda: all(
                level(c) >= MasteryLevel.L4_APPLY_WITH_GUIDANCE
                for c in capabilities)),
            (Maturity.S3_RELIABLE, lambda: all(
                len(results_by_capability[c][:3]) >= 3
                and all(r.passed for r in results_by_capability[c][:3])
                for c in capabilities)),
            (Maturity.S4_CORRECT, lambda: all(
                level(c) >= MasteryLevel.L5_APPLY_INDEPENDENTLY
                and any(r.passed and r.split in (Split.HIDDEN, Split.VALIDATION)
                        for r in results_by_capability[c])
                for c in capabilities)),
            (Maturity.S5_PROFICIENT, lambda: all(
                level(c) >= MasteryLevel.L7_GENERALIZES for c in capabilities)),
            (Maturity.S6_EXCELLENT, low_supervision),
            (Maturity.S7_FIELD_TRUSTED, lambda: any(
                level(c) >= MasteryLevel.L9_EXPERT for c in capabilities)),
        ]
        maturity = Maturity.S0_CREATED
        if capabilities:
            for stage, holds in ladder:
                if not holds():
                    break
                maturity = stage

        if maturity != profile.maturity:
            self.store.save_promotion(Promotion(
                agent_id=profile.id, from_maturity=profile.maturity,
                to_maturity=maturity))
            profile.maturity = maturity
            profile.updated_at = time.time()
            self.store.save_profile(profile)
        return maturity

    # -- field learning (document 05 section 6) -------------------------------
    def field_lesson(self, profile: AgentProfile, event: str,
                     rule_or_procedure: str, concept: str,
                     confidence: float = 0.5) -> ReusableLesson:
        """Real-world experience becomes a candidate, never validated by the
        single event that produced it - one deployed agent's experience must
        not silently corrupt every future agent's knowledge."""
        reusable = ReusableLesson(
            source_event=event, rule_or_procedure=rule_or_procedure,
            knowledge_class=KnowledgeClass.EXPERIENCE, confidence=confidence,
            validation_status=ValidationStatus.CANDIDATE,
            scope_domain=profile.domain, scope_concept=concept,
            source_agent_id=profile.id)
        self.store.save_reusable(reusable)
        self.store.record_metric("field.lesson", 1.0, agent_id=profile.id,
                                 detail=concept)
        return reusable

    # -- factory feedback (document 05 section 7) ------------------------------
    def metrics(self) -> Dict[str, Any]:
        profiles = self.store.profiles()
        all_results = []
        for p in profiles:
            all_results.extend(self.store.results(p.id, limit=1000))

        failure_counts = Counter(
            r.failure_mode for r in all_results if r.failure_mode and not r.passed)
        recurring_failure_modes = {k: v for k, v in failure_counts.items()
                                   if v > 1}

        preloaded_validated = [
            r.id for r in self.store.reusable_lessons(include_deprecated=False)
            if r.validation_status in (ValidationStatus.VALIDATED,
                                       ValidationStatus.SHARED)]

        predictive_tests: Dict[str, List[str]] = {}
        for p in profiles:
            for hidden in self.store.results(p.id, split=Split.HIDDEN,
                                             limit=1000):
                if not hidden.passed:
                    continue
                test = self.store.test(hidden.test_id)
                capability = test.capability if test else ""
                earlier = [
                    r.test_id for r in self.store.results(
                        p.id, capability=capability, split=Split.TRAINING,
                        limit=1000)
                    if r.at <= hidden.at]
                predictive_tests.setdefault(hidden.test_id, []).extend(earlier)

        return {
            "recurring_failure_modes": recurring_failure_modes,
            "preloaded_lessons_validated": preloaded_validated,
            "predictive_tests": predictive_tests,
        }
