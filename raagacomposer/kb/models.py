"""What the Knowledge Base is made of.

Knowledge Base architecture specification sections 4 to 13.  These are plain
records; the store persists them and the service does the work.

One departure from a literal reading of the specification is worth stating
here because everything else follows from it.  Section 4 defines a Knowledge
Item with a ``statement``; section 6 defines a Claim with subject, predicate
and object; section 33 lists both as tables, and lists ``entities``,
``procedures`` and ``examples`` besides.  Implemented literally that is four
identity spaces holding overlapping content, and a relationship (section 7,
whose endpoints are ``knowledge_id``) could not point at three of them.

So there is **one node table and one id space**.  A claim is a Knowledge Item
whose subject, predicate and value are filled in.  An entity is a Knowledge
Item of an entity type.  Procedures and examples are Knowledge Items with a
1:1 detail row carrying the extra fields they need.  The specification's table
names survive as read-only views over that one table, so the logical model it
asks for is all there and can be queried by those names - but there is exactly
one place a piece of knowledge lives, one kind of id, and nothing can drift
out of step with itself.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------
class KnowledgeType:
    """Section 5.  What kind of thing a Knowledge Item is."""

    ENTITY = "entity"                  # 5.1 raga, tala, instrument, composer
    FACT = "fact"                      # 5.2 arohanam, tala structure
    PROCEDURE = "procedure"            # 5.3 how to build an alapana
    PATTERN = "pattern"                # 5.4 prayogas, contours, cadences
    CONSTRAINT = "constraint"          # 5.5 avoidances, conditional rules
    EXAMPLE = "example"                # 5.6 a segment, a demonstration
    NEGATIVE = "negative"              # 5.7 useful failures
    PREFERENCE = "preference"          # 5.8 style, kept apart from fact
    META = "meta"                      # 5.9 "teachers disagree here"

    ALL = (ENTITY, FACT, PROCEDURE, PATTERN, CONSTRAINT, EXAMPLE, NEGATIVE,
           PREFERENCE, META)

    #: Types that assert something about the world and can therefore be
    #: contradicted.  A preference cannot be wrong, only different.
    ASSERTIVE = (FACT, PATTERN, CONSTRAINT, PROCEDURE)


class Status:
    """Section 11.  Where a piece of knowledge is in its life."""

    CANDIDATE = "candidate"
    LEARNED = "learned"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    NEEDS_REVIEW = "needs_review"

    ALL = (CANDIDATE, LEARNED, VALIDATED, ACCEPTED, DISPUTED, SUPERSEDED,
           REJECTED, DEPRECATED, NEEDS_REVIEW)
    #: Statuses that may be handed to composition or teaching.
    USABLE = (LEARNED, VALIDATED, ACCEPTED, DISPUTED)
    #: Statuses that are no longer the current answer.
    RETIRED = (SUPERSEDED, REJECTED, DEPRECATED)

    LABELS = {
        CANDIDATE: "Candidate", LEARNED: "Learned", VALIDATED: "Validated",
        ACCEPTED: "Accepted", DISPUTED: "Disputed", SUPERSEDED: "Superseded",
        REJECTED: "Rejected", DEPRECATED: "Deprecated",
        NEEDS_REVIEW: "Needs review",
    }


class Scope:
    """Section 20.  An item may belong to several."""

    GLOBAL = "GLOBAL"
    MUSIC = "MUSIC"
    CARNATIC = "CARNATIC"
    RAGA = "RAGA"
    TALA = "TALA"
    COMPOSITION = "COMPOSITION"
    SINGING = "SINGING"
    INSTRUMENTATION = "INSTRUMENTATION"
    CURRICULUM = "CURRICULUM"
    TRAINING = "TRAINING"
    USER_APPROVED = "USER_APPROVED"

    ALL = (GLOBAL, MUSIC, CARNATIC, RAGA, TALA, COMPOSITION, SINGING,
           INSTRUMENTATION, CURRICULUM, TRAINING, USER_APPROVED)


class Relation:
    """Section 7.  Relationships are first-class, not prose."""

    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEMONSTRATES = "demonstrates"
    EXAMPLE_OF = "example_of"
    PREREQUISITE_FOR = "prerequisite_for"
    SIMILAR_TO = "similar_to"
    DIFFERENT_FROM = "different_from"
    USED_IN = "used_in"
    GENERATED_FROM = "generated_from"
    TAUGHT_BY = "taught_by"
    BELONGS_TO_RAGA = "belongs_to_raga"
    BELONGS_TO_TALA = "belongs_to_tala"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    REFINES = "refines"
    SUPERSEDES = "supersedes"
    CONFLICTS_WITH = "conflicts_with"
    VERIFIED_BY = "verified_by"
    REJECTED_BY = "rejected_by"
    DEPENDS_ON = "depends_on"

    ALL = (IS_A, PART_OF, RELATED_TO, DERIVED_FROM, SUPPORTS, CONTRADICTS,
           DEMONSTRATES, EXAMPLE_OF, PREREQUISITE_FOR, SIMILAR_TO,
           DIFFERENT_FROM, USED_IN, GENERATED_FROM, TAUGHT_BY,
           BELONGS_TO_RAGA, BELONGS_TO_TALA, FOLLOWS, PRECEDES, REFINES,
           SUPERSEDES, CONFLICTS_WITH, VERIFIED_BY, REJECTED_BY, DEPENDS_ON)

    #: Reading a relationship the other way round, where that has a name.
    INVERSE = {
        PART_OF: "has_part", PREREQUISITE_FOR: "depends_on",
        DEPENDS_ON: PREREQUISITE_FOR, FOLLOWS: PRECEDES, PRECEDES: FOLLOWS,
        SUPERSEDES: "superseded_by", EXAMPLE_OF: DEMONSTRATES,
        DEMONSTRATES: EXAMPLE_OF,
    }
    #: Symmetric relations - traversal must not care which end it started at.
    SYMMETRIC = (RELATED_TO, SIMILAR_TO, DIFFERENT_FROM, CONFLICTS_WITH)


class ConflictState:
    """Section 12.  Music disagrees for good reasons; say which."""

    UNRESOLVED = "unresolved"
    CONTEXT_DEPENDENT = "context_dependent"
    RESOLVED_A = "resolved_for_claim_a"
    RESOLVED_B = "resolved_for_claim_b"
    BOTH_VALID = "both_valid_under_conditions"
    REJECTED_BOTH = "rejected_both"

    ALL = (UNRESOLVED, CONTEXT_DEPENDENT, RESOLVED_A, RESOLVED_B, BOTH_VALID,
           REJECTED_BOTH)
    OPEN = (UNRESOLVED, CONTEXT_DEPENDENT)


class ExtractionMethod:
    """How a piece of evidence came to exist - section 9's fourth question."""

    AUDIO = "audio_derived"
    TRANSCRIPT = "transcript_derived"
    INFERRED = "inferred"
    STRUCTURAL = "structural_library"
    USER = "user_supplied"
    EXPERIENCE = "application_experience"

    #: Methods where something was actually observed rather than reasoned.
    OBSERVED = (AUDIO, STRUCTURAL, USER)


#: Section 10's bands, as one place rather than scattered comparisons.
CONFIDENCE_BANDS = (
    (0.90, "Very high"), (0.75, "High"), (0.55, "Moderate"), (0.30, "Low"),
    (0.00, "Unverified"),
)


def confidence_band(value: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if value >= threshold:
            return label
    return "Unverified"


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------
@dataclass
class KnowledgeItem:
    """The smallest durable unit - sections 4 and 6 in one shape.

    ``statement`` is the human-readable explanation the specification asks for;
    ``structured_value`` is the machine-readable form beside it.  When the item
    is a claim, ``subject``/``predicate``/``object_value`` carry it in the
    form section 6 wants, and the same row is still a node relationships can
    point at.
    """

    knowledge_id: str = field(default_factory=lambda: new_id("kn"))
    canonical_name: str = ""
    knowledge_type: str = KnowledgeType.FACT
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    statement: str = ""
    structured_value: Dict[str, Any] = field(default_factory=dict)
    scope: List[str] = field(default_factory=lambda: [Scope.CARNATIC])
    status: str = Status.CANDIDATE
    confidence: float = 0.0
    #: How the confidence was arrived at, so section 41 can explain it.
    confidence_parts: Dict[str, float] = field(default_factory=dict)
    importance: float = 0.5
    source_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_verified_at: float = 0.0
    version: int = 1
    tags: List[str] = field(default_factory=list)

    # -- recommended fields (section 4) ---------------------------------
    valid_from: str = ""
    valid_until: str = ""
    language: str = ""
    difficulty: str = ""
    curriculum_level: str = ""
    owner_or_creator: str = ""
    review_state: str = ""
    usage_count: int = 0
    last_used_at: float = 0.0
    learned_by: str = ""
    notes: str = ""

    # -- musical shortcuts, indexed because everything filters on them ---
    raga: str = ""
    tala: str = ""

    @property
    def is_claim(self) -> bool:
        return bool(self.subject and self.predicate)

    @property
    def usable(self) -> bool:
        return self.status in Status.USABLE

    @property
    def band(self) -> str:
        return confidence_band(self.confidence)

    def display(self) -> str:
        return self.statement or (
            f"{self.subject} {self.predicate} {self.object_value}".strip())


@dataclass
class Relationship:
    """Section 7.  An edge, with its own evidence and confidence."""

    relationship_id: str = field(default_factory=lambda: new_id("rel"))
    source_knowledge_id: str = ""
    relation_type: str = Relation.RELATED_TO
    target_knowledge_id: str = ""
    confidence: float = 0.5
    evidence: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = Status.ACCEPTED


@dataclass
class Source:
    """Section 9.  Where something came from, as a thing in its own right."""

    source_id: str = field(default_factory=lambda: new_id("src"))
    source_type: str = ""
    title: str = ""
    author_or_channel: str = ""
    reference: str = ""              # URL or file reference
    published_date: str = ""
    acquired_date: float = field(default_factory=time.time)
    license_or_access_notes: str = ""
    language: str = ""
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Set when this source is a training run's source, so the two records
    #: can be joined without the KB owning the training tables.
    training_source_id: str = ""


@dataclass
class Evidence:
    """Section 9.  The specific thing in a source that supports a claim."""

    evidence_id: str = field(default_factory=lambda: new_id("ev"))
    source_id: str = ""
    knowledge_id: str = ""
    source_segment: str = ""
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    transcript_excerpt: str = ""
    feature_reference: str = ""
    strength: float = 0.5
    extraction_method: str = ExtractionMethod.INFERRED
    #: Which learning run produced it, so a report can be found again.
    run_id: str = ""
    created_at: float = field(default_factory=time.time)
    #: False when this evidence argues *against* the claim - section 6 wants
    #: contradicting evidence kept beside the supporting kind.
    supports: bool = True

    @property
    def observed(self) -> bool:
        return self.extraction_method in ExtractionMethod.OBSERVED


@dataclass
class Conflict:
    """Section 12.  Two claims that cannot both be simply true."""

    conflict_id: str = field(default_factory=lambda: new_id("cft"))
    claim_a: str = ""
    claim_b: str = ""
    source_a: str = ""
    source_b: str = ""
    confidence_a: float = 0.0
    confidence_b: float = 0.0
    resolution_status: str = ConflictState.UNRESOLVED
    reviewer: str = ""
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0

    @property
    def open(self) -> bool:
        return self.resolution_status in ConflictState.OPEN


@dataclass
class Version:
    """Section 13.  The old reading, kept, with why it changed."""

    version_id: str = field(default_factory=lambda: new_id("ver"))
    knowledge_id: str = ""
    version: int = 1
    snapshot: Dict[str, Any] = field(default_factory=dict)
    changed_at: float = field(default_factory=time.time)
    reason: str = ""
    caused_by_source_id: str = ""
    caused_by_run_id: str = ""
    changed_by: str = "system"


@dataclass
class ProcedureDetail:
    """Section 23.  The extra fields a procedure needs, keyed to its item."""

    knowledge_id: str = ""
    goal: str = ""
    prerequisites: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    optional_branches: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)


@dataclass
class ExampleDetail:
    """Section 24.  Examples are first-class and must stay traceable."""

    knowledge_id: str = ""
    concept_demonstrated: str = ""
    notation: str = ""
    swaras: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    quality: float = 0.5
    curriculum_stage: str = ""


@dataclass
class FailureLesson:
    """Section 25.  A mistake, so it is not rediscovered."""

    lesson_id: str = field(default_factory=lambda: new_id("les"))
    task: str = ""
    attempted_method: str = ""
    result: str = ""
    failure_reason: str = ""
    correction: str = ""
    related_knowledge: List[str] = field(default_factory=list)
    source_or_run: str = ""
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class KnowledgeGap:
    """Section 30.  What the system knows it does not know."""

    gap_id: str = field(default_factory=lambda: new_id("gap"))
    subject: str = ""
    missing_information: str = ""
    importance: float = 0.5
    reason: str = ""
    suggested_search: str = ""
    curriculum_dependency: str = ""
    status: str = "open"
    created_at: float = field(default_factory=time.time)
    closed_at: float = 0.0


@dataclass
class UserCorrection:
    """Section 28.  What a person decided, kept rather than applied and lost."""

    correction_id: str = field(default_factory=lambda: new_id("cor"))
    knowledge_id: str = ""
    action: str = ""                 # mark_correct, mark_incorrect, replace ...
    explanation: str = ""
    previous_value: str = ""
    new_value: str = ""
    previous_confidence: float = 0.0
    new_confidence: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class HealthReport:
    """Section 40."""

    total_items: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    sources: int = 0
    evidence: int = 0
    relationships: int = 0
    orphan_items: int = 0
    duplicate_candidates: int = 0
    unresolved_conflicts: int = 0
    knowledge_gaps: int = 0
    low_confidence_items: int = 0
    last_backup: float = 0.0
    last_integrity_check: float = 0.0
    integrity_ok: bool = True
    schema_version: int = 0
    initialized_at: float = 0.0

    def summary(self) -> str:
        return (f"{self.total_items} item(s), {self.sources} source(s), "
                f"{self.evidence} evidence record(s), "
                f"{self.relationships} relationship(s); "
                f"{self.unresolved_conflicts} unresolved conflict(s), "
                f"{self.knowledge_gaps} gap(s), "
                f"{self.low_confidence_items} weakly supported")
