"""The generic half of a Trainer (document 02 section 1B, document 03).

A domain trainer supplies lessons and tests; ``AdaptiveTrainer`` decides
which test to give next, when to raise difficulty, when to remediate, and
when a test has been beaten enough to retire.  None of this knows anything
about music, grammar, or any other domain - it only reasons about
``TestSpec``/``TestResult`` and the mastery ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .mastery import next_test_level
from .models import (AgentProfile, Lesson, Remediation, Split, TestLevel,
                     TestResult, TestSpec)
from .protocols import Trainer
from .store import FactoryStore


@dataclass
class TrainerPolicy:
    raise_after: int = 3
    calibration_tolerance: float = 0.2
    remediate_after_repeats: int = 2
    retire_after_beats: int = 3


class AdaptiveTrainer:
    """Document 03: the test ladder is adaptive, not fixed.  This class owns
    the *when*; the domain ``Trainer`` owns the *what*."""

    def __init__(self, store: FactoryStore, domain_trainer: Trainer,
                policy: TrainerPolicy = TrainerPolicy()) -> None:
        self.store = store
        self.domain_trainer = domain_trainer
        self.policy = policy
        # The last remediation offered per lesson, so the anti-repeat rule
        # (never the same test/instruction twice) has something to compare
        # against.  Disposable derived state (document 04 section 2): not
        # persisted, rebuilt as training proceeds.
        self._last_remediation: Dict[str, Remediation] = {}

    # -- choosing a test -----------------------------------------------------
    def _is_retired(self, test: TestSpec) -> bool:
        if test.retired:
            return True
        stored = self.store.test(test.id)
        return bool(stored and stored.retired)

    def choose_test(self, profile: AgentProfile, lesson: Lesson,
                    candidates: Sequence[TestSpec],
                    history: Sequence[TestResult]) -> Optional[TestSpec]:
        # A candidate is trusted for its own fields, but "retired" is a
        # store-side fact: a domain trainer's build_tests may keep returning
        # the same TestSpec object it always has, unaware that after_result
        # retired it in the store a moment ago.
        pool = [t for t in candidates
               if t.split != Split.HIDDEN and not self._is_retired(t)]
        if not pool:
            return None

        mastery = self.store.mastery(profile.id, lesson.concept)
        base_level = next_test_level(mastery)
        at_level = sorted(
            [r for r in history if r.lesson_id == lesson.id
             and r.level == base_level],
            key=lambda r: r.at)
        preferred_level = base_level
        if self.should_raise_difficulty(at_level):
            higher = min(int(base_level) + 1, int(TestLevel.T10_REAL_WORLD))
            if any(int(t.level) == higher for t in pool):
                preferred_level = TestLevel(higher)

        attempted_ids = {r.test_id for r in history}

        def rank(group: Sequence[TestSpec]) -> Optional[TestSpec]:
            if not group:
                return None
            unseen = [t for t in group if t.id not in attempted_ids]
            source = unseen or list(group)
            return max(source, key=lambda t: (t.novelty, t.difficulty))

        at_preferred = [t for t in pool if t.level == preferred_level]
        chosen = rank(at_preferred)
        if chosen is not None:
            return chosen
        at_base = [t for t in pool if t.level == base_level]
        chosen = rank(at_base)
        if chosen is not None:
            return chosen
        return rank(pool)

    def choose_hidden(self, profile: AgentProfile, capability: str,
                      candidates: Sequence[TestSpec]) -> Optional[TestSpec]:
        pool = [t for t in candidates
               if t.split == Split.HIDDEN and t.capability == capability
               and not self._is_retired(t)]
        if not pool:
            return None
        attempted = {r.test_id for r in
                    self.store.results(profile.id, capability=capability,
                                       split=Split.HIDDEN, limit=1000)}
        unseen = [t for t in pool if t.id not in attempted]
        source = unseen or pool
        return max(source, key=lambda t: (t.novelty, t.difficulty, int(t.level)))

    # -- adapting --------------------------------------------------------
    def should_raise_difficulty(self, history_at_level: Sequence[TestResult]) -> bool:
        ordered = sorted(history_at_level, key=lambda r: r.at)
        recent = ordered[-self.policy.raise_after:]
        if len(recent) < self.policy.raise_after:
            return False
        return all(r.passed and r.calibrated for r in recent)

    def should_remediate(self, recent: Sequence[TestResult]) -> bool:
        ordered = sorted(recent, key=lambda r: r.at)
        if not ordered:
            return False
        last = ordered[-1]
        if not last.passed and last.student_confidence >= 0.7:
            return True
        failures = [r for r in ordered if not r.passed]
        window = failures[-self.policy.remediate_after_repeats:]
        if len(window) >= self.policy.remediate_after_repeats:
            modes = {r.failure_mode for r in window if r.failure_mode}
            if len(modes) == 1:
                return True
        return False

    def after_result(self, profile: AgentProfile, lesson: Lesson,
                     test: TestSpec, result: TestResult) -> Optional[Remediation]:
        self.domain_trainer.learn_from(result)

        if result.passed and not test.retired:
            history = self.store.results(profile.id, limit=1000)
            beats = sum(1 for r in history if r.test_id == test.id and r.passed)
            if beats >= self.policy.retire_after_beats:
                self.store.retire_test(test.id)
                self.store.record_metric("test.retired", 1.0, agent_id=profile.id,
                                         detail=test.id)
                self.domain_trainer.build_tests(lesson, profile, history)

        recent = self.store.results(profile.id, capability=lesson.concept,
                                    level=test.level, limit=20)

        remediation: Optional[Remediation] = None
        if self.should_remediate(recent):
            failures = [r for r in recent if not r.passed] or list(recent)
            remediation = self.domain_trainer.remediate(profile, lesson, failures)
            previous = self._last_remediation.get(lesson.id)
            if (previous is not None and previous.kind == remediation.kind
                   and previous.detail == remediation.detail):
                remediation = Remediation(
                    kind="level_down", lesson_id=lesson.id,
                    detail=remediation.detail or "repeated remediation",
                    next_level=remediation.next_level)
            self._last_remediation[lesson.id] = remediation
            self.store.record_metric("remediation", 1.0, agent_id=profile.id,
                                     detail=f"{lesson.concept}:{remediation.kind}")
        elif self.should_raise_difficulty(recent):
            self.store.record_metric("difficulty.raised", 1.0, agent_id=profile.id,
                                     detail=lesson.concept)

        return remediation
