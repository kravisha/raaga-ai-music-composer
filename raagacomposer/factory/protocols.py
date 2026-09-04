"""The three roles (document 02) as protocols.

An adapter for a domain is anything with these methods.  The core never
imports a domain; the domain imports the core.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, runtime_checkable

from .models import (AgentProfile, Dispute, KnowledgeClass, Lesson,
                     Performance, Reiteration, ReiterationCheck, Remediation,
                     Ruling, TestResult, TestSpec)


@runtime_checkable
class Student(Protocol):
    """Document 02 section 1A.  Learns the capability and does the work."""

    @property
    def profile(self) -> AgentProfile: ...

    def acquire(self, lesson: Lesson) -> None:
        """Step 1: receive or retrieve the lesson."""

    def reiterate(self, lesson: Lesson) -> Reiteration:
        """Steps 2 to 5 and R1 to R7: restate, explain, connect, example,
        counterexample, apply, self-check - in the student's own words,
        from what it actually knows."""

    def perform(self, test: TestSpec) -> Performance:
        """Do the test.  The claim is the student's own verdict on its
        answer; the confidence is how sure it is.  Uncertainty is exposed,
        not hidden."""

    def apply_correction(self, correction: str, lesson: Optional[Lesson]) -> None:
        """Take a ruling's correction on board."""


@runtime_checkable
class Trainer(Protocol):
    """Document 02 section 1B.  Teaches, challenges, measures, adapts, and
    learns while doing so."""

    def next_lesson(self, profile: AgentProfile,
                    history: Sequence[TestResult]) -> Optional[Lesson]:
        """The next lesson for this student, given what it has shown.
        None means the curriculum is complete."""

    def check_reiteration(self, lesson: Lesson,
                          reiteration: Reiteration) -> ReiterationCheck:
        """Did the restatement, explanation and examples hold up against
        the source knowledge?"""

    def build_tests(self, lesson: Lesson, profile: AgentProfile,
                    history: Sequence[TestResult]) -> List[TestSpec]:
        """Tests for this lesson at the level the student has earned, with
        novelty the student has not seen.  The adaptive trainer decides
        which to give; this supplies them."""

    def grade(self, test: TestSpec, performance: Performance) -> TestResult:
        """Score the performance and state the trainer's own claim and
        confidence, so a disagreement with the student is visible."""

    def remediate(self, profile: AgentProfile, lesson: Lesson,
                  failures: Sequence[TestResult]) -> Remediation:
        """Change the instruction or the practice.  Never the same test
        again."""

    def learn_from(self, result: TestResult) -> None:
        """The trainer is also a learner: retire beaten tests, note which
        tests predict, raise difficulty where earned."""


@runtime_checkable
class Rule(Protocol):
    """How hard knowledge reaches the Judge.  A rule either settles a
    dispute or stays silent."""

    @property
    def name(self) -> str: ...

    @property
    def knowledge_class(self) -> KnowledgeClass: ...

    def applies(self, dispute: Dispute) -> bool: ...

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        """A ruling, or None when this rule cannot settle it."""
