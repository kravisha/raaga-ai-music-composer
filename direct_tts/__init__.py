"""Manifest validation and dataset export planning for direct SPB TTS fine-tuning.

Dependency-light by design: only the standard library. No audio decoding,
no model loading, no training. See docs/spec/SPB_DIRECT_TTS_ADAPTATION.md
for the data and training contract this module enforces.
"""

from direct_tts.manifest import (
    ADMITTED_TARGET,
    REVIEWED,
    ValidationIssue,
    validate_entry,
    find_cross_entry_issues,
)
from direct_tts.planner import (
    ExportPlan,
    EvalEntryPlan,
    TrainEntryPlan,
    SESSION_DISJOINT,
    WITHIN_SESSION_BLOCKED,
    plan_export,
)
from direct_tts.verifier import (
    EntryVerification,
    VerificationResult,
    sha256_of_file,
    verify_plan,
)

__all__ = [
    "ADMITTED_TARGET",
    "REVIEWED",
    "ValidationIssue",
    "validate_entry",
    "find_cross_entry_issues",
    "ExportPlan",
    "EvalEntryPlan",
    "TrainEntryPlan",
    "SESSION_DISJOINT",
    "WITHIN_SESSION_BLOCKED",
    "plan_export",
    "EntryVerification",
    "VerificationResult",
    "sha256_of_file",
    "verify_plan",
]
