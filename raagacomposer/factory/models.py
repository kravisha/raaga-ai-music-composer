"""The framework's data objects (handoff document 06, "minimum data objects",
widened by documents 01 to 05).

Everything here is plain data.  The ladders are enums so a level is a value
that can be compared and stored, not a string that can be misspelt.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# ladders
# --------------------------------------------------------------------------
class MasteryLevel(IntEnum):
    """Document 01 section 5.  Partial mastery is per concept."""
    L0_UNKNOWN = 0
    L1_EXPOSED = 1
    L2_CAN_RESTATE = 2
    L3_CAN_EXPLAIN = 3
    L4_APPLY_WITH_GUIDANCE = 4
    L5_APPLY_INDEPENDENTLY = 5
    L6_DETECTS_ERRORS = 6
    L7_GENERALIZES = 7
    L8_CAN_TEACH = 8
    L9_EXPERT = 9

    @property
    def label(self) -> str:
        return self.name[3:].replace("_", " ").lower()


class TestLevel(IntEnum):
    """Document 03 section 2, the test ladder."""
    T0_RECOGNITION = 0
    T1_RECALL = 1
    T2_EXPLANATION = 2
    T3_CONTROLLED_APPLICATION = 3
    T4_INDEPENDENT_APPLICATION = 4
    T5_VARIATION = 5
    T6_ERROR_DETECTION = 6
    T7_CORRECTION = 7
    T8_GENERALIZATION = 8
    T9_ADVERSARIAL = 9
    T10_REAL_WORLD = 10

    @property
    def label(self) -> str:
        return self.name[self.name.index("_") + 1:].replace("_", " ").lower()


class Maturity(IntEnum):
    """Document 05 section 4, the maturity pipeline."""
    S0_CREATED = 0
    S1_EFFECTIVE = 1
    S2_WORKING = 2
    S3_RELIABLE = 3
    S4_CORRECT = 4
    S5_PROFICIENT = 5
    S6_EXCELLENT = 6
    S7_FIELD_TRUSTED = 7

    @property
    def label(self) -> str:
        return self.name[3:].replace("_", " ").lower()


class KnowledgeClass(str, Enum):
    """Document 04 section 1.  Hard knowledge decides disputes; everything
    else is defeasible."""
    HARD = "hard"
    HEURISTIC = "heuristic"
    EXPERIENCE = "experience"
    PROCEDURE = "procedure"
    TEST = "test"
    DISPUTE_LESSON = "dispute_lesson"


class Split(str, Enum):
    """Document 03 section 6, the anti-gaming rule.  Hidden tests are never
    practised on; regression holds tests the student has beaten."""
    TRAINING = "training"
    VALIDATION = "validation"
    HIDDEN = "hidden"
    REAL_WORLD = "real_world"
    REGRESSION = "regression"


class DisputeStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"


class ValidationStatus(str, Enum):
    """Document 04 section 5: raw event -> candidate -> validated -> shared."""
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    SHARED = "shared"
    DEPRECATED = "deprecated"


# --------------------------------------------------------------------------
# the agent
# --------------------------------------------------------------------------
@dataclass
class AgentSpec:
    """Document 05 section 1, what a new agent begins from."""
    name: str
    role: str
    domain: str
    mission: str = ""
    capabilities: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    input_contract: str = ""
    output_contract: str = ""
    success_metrics: Dict[str, float] = field(default_factory=dict)
    safety_constraints: List[str] = field(default_factory=list)
    environment: str = "factory"
    # Release-gate statements (document 05 section 5): the deployer says
    # these exist; the gate checks they are stated, not that they work.
    rollback: str = ""
    permissions: List[str] = field(default_factory=list)
    monitoring: str = ""
    escalation: str = ""


@dataclass
class AgentProfile:
    """Document 06 AgentProfile.  Strengths and weaknesses are per concept."""
    id: str = field(default_factory=lambda: new_id("agent"))
    name: str = ""
    role: str = ""
    domain: str = ""
    capabilities: List[str] = field(default_factory=list)
    maturity: Maturity = Maturity.S0_CREATED
    current_curriculum: str = ""
    current_lesson_id: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    knowledge_version: str = ""
    spec: Optional[AgentSpec] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# lessons and reiteration
# --------------------------------------------------------------------------
@dataclass
class Lesson:
    """Document 01 section 3, a learning unit.  ``mastery`` is not here: it
    belongs to an agent and a concept (MasteryRecord), because one lesson is
    taught to many agents."""
    id: str = field(default_factory=lambda: new_id("lesson"))
    domain: str = ""
    concept: str = ""
    objective: str = ""
    prerequisites: List[str] = field(default_factory=list)   # concept names
    source_knowledge: List[str] = field(default_factory=list)  # provenance
    explanation: str = ""
    examples: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    practice_tasks: List[str] = field(default_factory=list)
    test_tasks: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    common_errors: List[str] = field(default_factory=list)
    remediation: List[str] = field(default_factory=list)
    knowledge_class: KnowledgeClass = KnowledgeClass.HARD
    confidence: float = 1.0
    version: int = 1
    # The domain object this lesson was built from, if any (a curriculum
    # unit id, a rule name), so the adapter can find its way back.
    origin: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reiteration:
    """Document 04 section 3, R1 to R8, as the student produced them."""
    lesson_id: str = ""
    agent_id: str = ""
    restate: str = ""
    explain: str = ""
    connect: str = ""
    example: str = ""
    counterexample: str = ""
    apply_summary: str = ""
    apply_score: float = 0.0
    self_check: str = ""
    retest_due_at: float = 0.0
    at: float = field(default_factory=time.time)


@dataclass
class ReiterationCheck:
    """The trainer's verdict on a reiteration: which parts held up."""
    restate_ok: bool = False
    explain_ok: bool = False
    connect_ok: bool = False
    example_ok: bool = False
    counterexample_ok: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.restate_ok and self.explain_ok and self.example_ok


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
@dataclass
class TestSpec:
    """Document 06 Test plus document 03 section 4's quality metrics."""
    id: str = field(default_factory=lambda: new_id("test"))
    capability: str = ""           # concept under test
    level: TestLevel = TestLevel.T1_RECALL
    novelty: float = 0.0           # 0 seen before .. 1 never seen
    difficulty: float = 0.5        # 0 .. 1
    ambiguity: float = 0.0         # 0 objective .. 1 subjective
    objective: bool = True
    expected: str = ""
    acceptable_range: str = ""
    failure_mode_targeted: str = ""
    split: Split = Split.TRAINING
    seed: int = 0
    retired: bool = False
    version: int = 1
    author_agent_id: str = ""      # set when an agent wrote the test (L8)
    lesson_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class Performance:
    """What the student hands back for a test."""
    output: str = ""
    claim: str = ""                # the student's own verdict or answer
    confidence: float = 0.5        # how sure the student is of its claim
    evidence: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    id: str = field(default_factory=lambda: new_id("result"))
    test_id: str = ""
    agent_id: str = ""
    lesson_id: str = ""
    level: TestLevel = TestLevel.T1_RECALL
    split: Split = Split.TRAINING
    score: float = 0.0
    passed: bool = False
    student_claim: str = ""
    student_confidence: float = 0.0
    trainer_claim: str = ""
    trainer_confidence: float = 0.0
    failure_mode: str = ""
    evidence: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    judge_needed: bool = False
    dispute_id: str = ""
    at: float = field(default_factory=time.time)

    @property
    def calibrated(self) -> bool:
        """Confidence within 0.2 of the score (document 03 section 3)."""
        return abs(self.student_confidence - self.score) <= 0.2


@dataclass
class Remediation:
    """Document 03 section 3: what the trainer changes after failure.  A
    remediation that changes nothing is not one."""
    kind: str = ""                 # guided | different_practice | level_down | reteach
    lesson_id: str = ""
    detail: str = ""
    next_level: Optional[TestLevel] = None
    payload: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# disputes and rulings
# --------------------------------------------------------------------------
@dataclass
class Dispute:
    """Document 02 section 2, what the Judge receives."""
    id: str = field(default_factory=lambda: new_id("dispute"))
    agent_id: str = ""
    test_id: str = ""
    lesson_id: str = ""
    question: str = ""
    student_claim: str = ""
    trainer_claim: str = ""
    evidence_student: List[str] = field(default_factory=list)
    evidence_trainer: List[str] = field(default_factory=list)
    student_confidence: float = 0.0
    trainer_confidence: float = 0.0
    shared_knowledge: List[str] = field(default_factory=list)
    applicable_rules: List[str] = field(default_factory=list)
    status: DisputeStatus = DisputeStatus.OPEN
    ruling_id: str = ""
    critical: bool = False
    at: float = field(default_factory=time.time)


@dataclass
class Ruling:
    """Document 02 section 3, what the Judge returns."""
    id: str = field(default_factory=lambda: new_id("ruling"))
    dispute_id: str = ""
    ruling: str = ""               # accepted text, or "unresolved"
    accepted_claim: str = ""       # "student" | "trainer" | "neither" | ""
    rejected_claim: str = ""
    rationale: str = ""
    confidence: float = 0.0
    unresolved_issues: List[str] = field(default_factory=list)
    correction_student: str = ""
    correction_trainer: str = ""
    reusable_lesson_id: str = ""
    needs_external_evidence: bool = False
    decided_by: str = ""           # rule name, "escalation", or ""
    at: float = field(default_factory=time.time)

    @property
    def resolved(self) -> bool:
        return self.ruling != "" and self.ruling != "unresolved"


@dataclass
class ReusableLesson:
    """Document 06 ReusableLesson with document 04 section 6's deprecation
    fields.  Moves candidate -> validated -> shared; never silently
    accumulates a wrong rule."""
    id: str = field(default_factory=lambda: new_id("reuse"))
    source_event: str = ""         # a ruling id, a result id, a field event
    rule_or_procedure: str = ""
    knowledge_class: KnowledgeClass = KnowledgeClass.DISPUTE_LESSON
    confidence: float = 0.5
    validation_status: ValidationStatus = ValidationStatus.CANDIDATE
    scope_domain: str = ""
    scope_concept: str = ""
    source_agent_id: str = ""
    version: int = 1
    validations: int = 0
    last_validated_at: float = 0.0
    superseded_by: str = ""
    deprecated: bool = False
    created_at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------
# mastery, promotion, gates
# --------------------------------------------------------------------------
@dataclass
class MasteryRecord:
    agent_id: str = ""
    concept: str = ""
    level: MasteryLevel = MasteryLevel.L0_UNKNOWN
    evidence: List[str] = field(default_factory=list)   # result/ruling ids
    failures_at_level: int = 0
    updated_at: float = field(default_factory=time.time)


@dataclass
class Promotion:
    id: str = field(default_factory=lambda: new_id("promo"))
    agent_id: str = ""
    from_maturity: Maturity = Maturity.S0_CREATED
    to_maturity: Maturity = Maturity.S0_CREATED
    evidence: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class GateReport:
    gate: str = ""                 # "promotion" | "release"
    passed: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def failed_checks(self) -> List[str]:
        return [name for name, ok in self.checks.items() if not ok]
