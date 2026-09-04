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
from .store import FactoryStore
from .mastery import apply_evidence, next_test_level
from .trainer import AdaptiveTrainer, TrainerPolicy
from .judge import convene, should_convene
from .cycle import CycleOutcome, LearningCycle
from .gates import GateThresholds, promotion_gate, release_gate
from .factory import AgentFactory

__all__ = [
    "AgentProfile", "AgentSpec", "Dispute", "DisputeStatus", "GateReport",
    "KnowledgeClass", "Lesson", "MasteryLevel", "MasteryRecord", "Maturity",
    "Performance", "Promotion", "Reiteration", "ReiterationCheck",
    "Remediation", "ReusableLesson", "Ruling", "Split", "TestLevel",
    "TestResult", "TestSpec", "ValidationStatus", "Rule", "Student", "Trainer",
    "FactoryStore", "apply_evidence", "next_test_level", "AdaptiveTrainer",
    "TrainerPolicy", "convene", "should_convene", "CycleOutcome",
    "LearningCycle", "GateThresholds", "promotion_gate", "release_gate",
    "AgentFactory",
]
