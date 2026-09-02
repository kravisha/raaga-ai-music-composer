"""The things the Training tab moves around.

Training specification sections 5, 6, 8 and 9.  These are plain records with
no behaviour beyond describing themselves: the services do the work, the store
persists them, and the UI renders them.  Keeping them dumb is what lets a new
kind of source arrive later without the queue, the report or the tab noticing.

The vocabularies below - accessibility, run status, objective status - are the
specification's own, spelled as constants so a typo is an import error rather
than a row that never matches a filter.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# vocabularies (specification sections 3.3, 5, 6)
# --------------------------------------------------------------------------
class Accessibility:
    """What of a source we can actually reach - section 4."""

    ACCESSIBLE = "accessible"
    TRANSCRIPT = "transcript_available"
    METADATA_ONLY = "metadata_only"
    USER_FILE_REQUIRED = "user_file_required"
    NOT_ACCESSIBLE = "not_accessible"
    UNSUPPORTED = "unsupported"
    ALREADY_LEARNED = "already_learned"

    #: The states from which content can actually be analysed.
    ANALYSABLE = (ACCESSIBLE, TRANSCRIPT)

    LABELS = {
        ACCESSIBLE: "Accessible",
        TRANSCRIPT: "Transcript available",
        METADATA_ONLY: "Metadata only",
        USER_FILE_REQUIRED: "User file required",
        NOT_ACCESSIBLE: "Not accessible",
        UNSUPPORTED: "Unsupported",
        ALREADY_LEARNED: "Already learned",
    }


class RunStatus:
    """Every state a source passes through - section 3.3."""

    QUEUED = "queued"
    CHECKING_ACCESS = "checking_access"
    FETCHING_METADATA = "fetching_metadata"
    ACQUIRING_TRANSCRIPT = "acquiring_transcript"
    PREPARING_CONTENT = "preparing_content"
    ANALYZING = "analyzing"
    EXTRACTING_KNOWLEDGE = "extracting_knowledge"
    VALIDATING = "validating"
    AWAITING_REVIEW = "awaiting_review"
    SAVING_KNOWLEDGE = "saving_knowledge"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    SOURCE_INACCESSIBLE = "source_inaccessible"

    #: Nothing further will happen to a run in one of these.
    TERMINAL = (COMPLETED, SKIPPED, FAILED, SOURCE_INACCESSIBLE)
    #: A run interrupted mid-flight is returned to the queue on restart.
    IN_FLIGHT = (CHECKING_ACCESS, FETCHING_METADATA, ACQUIRING_TRANSCRIPT,
                 PREPARING_CONTENT, ANALYZING, EXTRACTING_KNOWLEDGE,
                 VALIDATING, SAVING_KNOWLEDGE)

    LABELS = {
        QUEUED: "Queued",
        CHECKING_ACCESS: "Checking access",
        FETCHING_METADATA: "Fetching metadata",
        ACQUIRING_TRANSCRIPT: "Acquiring transcript",
        PREPARING_CONTENT: "Preparing content",
        ANALYZING: "Analyzing",
        EXTRACTING_KNOWLEDGE: "Extracting knowledge",
        VALIDATING: "Validating",
        AWAITING_REVIEW: "Awaiting review",
        SAVING_KNOWLEDGE: "Saving knowledge",
        COMPLETED: "Completed",
        SKIPPED: "Skipped",
        FAILED: "Failed",
        SOURCE_INACCESSIBLE: "Source inaccessible",
    }


class ObjectiveStatus:
    """How an objective came out - section 6."""

    NOT_STARTED = "not_started"
    PARTIAL = "partially_learned"
    LEARNED = "learned"
    NOT_PRESENT = "not_present_in_source"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"

    LABELS = {
        NOT_STARTED: "Not started",
        PARTIAL: "Partially learned",
        LEARNED: "Learned",
        NOT_PRESENT: "Not present in source",
        UNCERTAIN: "Uncertain",
        NEEDS_REVIEW: "Needs review",
    }


class KnowledgeStatus:
    ACTIVE = "active"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass
class SearchQuery:
    """What the creator typed, and how they narrowed it - section 3.1."""

    phrase: str = ""
    max_results: int = 10
    source_filter: str = ""          # "" = every provider
    content_type: str = ""
    difficulty: str = ""
    language: str = ""
    duration_preference: str = ""    # "", short, medium, long
    include_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase, "max_results": self.max_results,
            "source_filter": self.source_filter,
            "content_type": self.content_type, "difficulty": self.difficulty,
            "language": self.language,
            "duration_preference": self.duration_preference,
            "include_keywords": list(self.include_keywords),
            "exclude_keywords": list(self.exclude_keywords),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        return cls(**{k: v for k, v in (data or {}).items()
                      if k in cls.__dataclass_fields__})


@dataclass
class LearningSource:
    """One normalised search result - specification section 5.

    Every provider produces this shape, whatever it searched, which is what
    lets the Training tab stay unchanged when a new provider is added.
    """

    source_id: str = field(default_factory=lambda: new_id("src"))
    source_type: str = ""            # video, article, transcript, local_file
    title: str = ""
    url: str = ""
    author: str = ""
    description: str = ""
    duration: float = 0.0            # seconds, 0 when unknown
    published_date: str = ""
    language: str = ""
    relevance_score: float = 0.0
    accessibility_status: str = Accessibility.METADATA_ONLY
    transcript_available: bool = False
    previously_learned: bool = False
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Set when the creator supplies the file the source could not give us.
    local_path: str = ""
    search_id: str = ""
    found_at: float = field(default_factory=time.time)

    @property
    def can_be_analysed(self) -> bool:
        return (bool(self.local_path)
                or self.accessibility_status in Accessibility.ANALYSABLE)

    @property
    def duration_label(self) -> str:
        if self.duration <= 0:
            return "-"
        minutes, seconds = divmod(int(self.duration), 60)
        return f"{minutes}:{seconds:02d}"


@dataclass
class Objective:
    """Something we mean to get out of a source - section 6."""

    objective_id: str = field(default_factory=lambda: new_id("obj"))
    run_id: str = ""
    description: str = ""
    category: str = "general"
    priority: int = 2                # 1 highest
    status: str = ObjectiveStatus.NOT_STARTED
    confidence: float = 0.0
    evidence: str = ""
    outcome: str = ""
    user_defined: bool = False
    position: int = 0

    @property
    def met(self) -> bool:
        return self.status in (ObjectiveStatus.LEARNED, ObjectiveStatus.PARTIAL)


@dataclass
class KnowledgeEntry:
    """One learned thing, and everything needed to trace it - section 9.

    ``source_id``, ``source_url``, ``source_title``, ``run_id``,
    ``objective_id`` and ``source_timestamp`` together answer every question
    section 16 requires of a learned item.  None of them is optional in
    practice: an entry that cannot say where it came from should not be here.
    """

    knowledge_id: str = field(default_factory=lambda: new_id("kn"))
    subject: str = ""
    concept: str = ""
    normalized_statement: str = ""
    category: str = ""
    raga: str = ""
    tala: str = ""
    difficulty: str = ""
    source_id: str = ""
    source_url: str = ""
    source_title: str = ""
    source_timestamp: str = ""
    evidence: str = ""
    confidence: float = 0.5
    date_learned: float = field(default_factory=time.time)
    related_knowledge: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: int = 1
    status: str = KnowledgeStatus.ACTIVE
    run_id: str = ""
    objective_id: str = ""
    user_approved: bool = False
    contradicted: bool = False

    def fingerprint_values(self) -> List[str]:
        return [self.subject.lower().strip(), self.category.lower().strip(),
                self.raga.lower().strip(),
                self.normalized_statement.lower().strip()]


@dataclass
class Conflict:
    """A source disagreeing with what we already hold - section 8.7."""

    conflict_id: str = field(default_factory=lambda: new_id("cft"))
    run_id: str = ""
    knowledge_id: str = ""
    existing_claim: str = ""
    new_claim: str = ""
    source_evidence: str = ""
    existing_confidence: float = 0.0
    new_confidence: float = 0.0
    recommendation: str = ""
    resolved: bool = False
    resolution: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class LearningRun:
    """One pass over one source - the unit the queue moves."""

    run_id: str = field(default_factory=lambda: new_id("run"))
    source_id: str = ""
    search_phrase: str = ""
    position: int = 0
    status: str = RunStatus.QUEUED
    progress: float = 0.0
    detail: str = ""
    result: str = ""
    error: str = ""
    attempts: int = 0
    queued_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    #: A relearn points back at the run it supersedes rather than erasing it.
    supersedes: str = ""

    @property
    def finished(self) -> bool:
        return self.status in RunStatus.TERMINAL


@dataclass
class LearningReport:
    """What section 8 requires of every completed source.

    ``understood`` and ``learned`` are deliberately separate fields rather than
    one narrative: rule 5 of section 20 makes that distinction non-negotiable,
    and a single field would quietly collapse it.
    """

    run_id: str = ""
    source: Optional[LearningSource] = None
    generated_at: float = field(default_factory=time.time)
    objectives: List[Objective] = field(default_factory=list)
    summary: str = ""
    understood: str = ""
    learned: List[str] = field(default_factory=list)
    confirmed: List[str] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    practical_application: List[str] = field(default_factory=list)
    confidence: float = 0.0
    next_learning: List[str] = field(default_factory=list)
    knowledge_ids: List[str] = field(default_factory=list)
    analysed_representation: str = ""
    honest_limits: List[str] = field(default_factory=list)

    @property
    def objectives_met(self) -> int:
        return sum(1 for o in self.objectives if o.met)

    def confidence_band(self) -> str:
        if self.confidence >= 0.7:
            return "High"
        return "Medium" if self.confidence >= 0.4 else "Low"
