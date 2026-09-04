"""The Agent Factory: a domain-independent way to make an agent that learns,
proves what it learned, is corrected, and improves before it is deployed.

Specification: docs/spec/agent_factory/ (Universal Learning Framework v0.1).
Plan and decisions: docs/PLAN_agent_factory.md.

Nothing in this package knows about music.  The Raga agent is the first
Student (raagacomposer/agent/student.py) with its own Trainer
(raagacomposer/agent/trainer.py); a second domain plugs in the same way.
"""
from .models import (AgentProfile, AgentSpec, Dispute, DisputeStatus,
                     GateReport, KnowledgeClass, Lesson, MasteryLevel,
                     MasteryRecord, Maturity, Performance, Promotion,
                     Reiteration, ReiterationCheck, Remediation,
                     ReusableLesson, Ruling, Split, TestLevel, TestResult,
                     TestSpec, ValidationStatus)
from .protocols import Rule, Student, Trainer

__all__ = [
    "AgentProfile", "AgentSpec", "Dispute", "DisputeStatus", "GateReport",
    "KnowledgeClass", "Lesson", "MasteryLevel", "MasteryRecord", "Maturity",
    "Performance", "Promotion", "Reiteration", "ReiterationCheck",
    "Remediation", "ReusableLesson", "Ruling", "Split", "TestLevel",
    "TestResult", "TestSpec", "ValidationStatus", "Rule", "Student", "Trainer",
]
