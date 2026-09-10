"""Dry-run dataset export planning.

Given a list of manifest entries (plain dicts, as loaded from JSON), decide
which ones are admitted into a train split and which into an eval split,
refusing on any contract violation. This module never opens an audio file,
never loads a model and never starts training: it only computes an export
plan (which reviewed clip would go where, under what new export path).

The plan this module produces is a *metadata* plan: it reasons entirely
about the manifest dict fields (paths, hashes, review flags, intervals).
``ExportPlan.metadata_ready`` reflects only that check. It is deliberately
not called ``ready``, because passing it does not mean the referenced
source/clip files actually exist on disk or that their bytes still match
the recorded hashes -- that requires reading real files, which is what
``direct_tts.verifier.verify_plan`` does as a separate, explicit step.
"""

from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from direct_tts.manifest import (
    EVAL,
    TRAIN,
    ValidationIssue,
    find_cross_entry_issues,
    validate_entry,
)

DEFAULT_EXPORT_ROOT = "dataset"

SESSION_DISJOINT = "session_disjoint"
WITHIN_SESSION_BLOCKED = "within_session_blocked"
_EVAL_POLICIES = frozenset({SESSION_DISJOINT, WITHIN_SESSION_BLOCKED})


@dataclass(frozen=True)
class TrainEntryPlan:
    id: str
    export_path: str
    source_path: str
    source_sha256: str
    reviewed_clip_path: str
    reviewed_clip_sha256: str
    start_seconds: float
    end_seconds: float
    text: str
    split: str = TRAIN


@dataclass(frozen=True)
class EvalEntryPlan:
    id: str
    export_path: str
    source_path: str
    source_sha256: str
    reviewed_clip_path: str
    reviewed_clip_sha256: str
    start_seconds: float
    end_seconds: float
    text: str
    eval_kind: str  # "session_disjoint" or "within_session_blocked"
    split: str = EVAL


@dataclass(frozen=True)
class RejectedEntry:
    id: str
    reasons: list[str]


@dataclass(frozen=True)
class ExcludedEntry:
    id: str
    reason: str


@dataclass(frozen=True)
class AnchorEntry:
    id: str
    source_path: str | None


@dataclass(frozen=True)
class ExportPlan:
    train: list[TrainEntryPlan] = field(default_factory=list)
    eval: list[EvalEntryPlan] = field(default_factory=list)
    diagnostic_anchors: list[AnchorEntry] = field(default_factory=list)
    excluded: list[ExcludedEntry] = field(default_factory=list)
    rejected: list[RejectedEntry] = field(default_factory=list)
    eval_policy: str = SESSION_DISJOINT
    metadata_ready: bool = False
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def d(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: d(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [d(v) for v in obj]
            return obj

        return d(self)


def _export_path(export_root: str, split: str, entry: Mapping[str, Any]) -> str:
    entry_id = str(entry.get("id"))
    _, ext = posixpath.splitext(str(entry.get("path") or ""))
    ext = ext or ".wav"
    return posixpath.join(export_root, split, f"{entry_id}{ext}")


def _is_within_root(export_root: str, export_path: str) -> bool:
    """Defense in depth: confirm the resolved export path cannot escape the
    requested root, even though validate_entry already restricts ids to
    safe filename stems with no path separators."""
    root_norm = posixpath.normpath(export_root)
    path_norm = posixpath.normpath(export_path)
    return path_norm == root_norm or path_norm.startswith(root_norm + "/")


def plan_export(
    entries: Sequence[Mapping[str, Any]],
    export_root: str = DEFAULT_EXPORT_ROOT,
    exists_fn: Callable[[str], bool] = os.path.exists,
    eval_policy: str = SESSION_DISJOINT,
    dry_run_view: bool = False,
) -> ExportPlan:
    """Compute a metadata export plan.

    ``eval_policy`` controls whether eval entries drawn from the same
    ``source_group`` as a train entry are allowed to satisfy readiness:

    - ``"session_disjoint"`` (default): at least one eval entry must be
      independent of every train source group. Within-session eval entries
      are still reported (truthfully labelled ``within_session_blocked``)
      but do not by themselves satisfy readiness.
    - ``"within_session_blocked"``: an explicit, narrower allowance for a
      bounded engineering pilot. Within-session eval entries may satisfy
      readiness. Entries are still truthfully labelled by their actual
      session relationship (never relabelled as session-disjoint), and
      this never implies the result generalizes across sessions.

    ``dry_run_view`` opts out of the "export root must not already exist"
    check below, for callers that only want to preview a plan against a
    root that may already hold prior output. It has no effect on any other
    check, and never makes the plan execution-ready by itself.
    """
    if eval_policy not in _EVAL_POLICIES:
        raise ValueError(
            f"eval_policy must be one of {sorted(_EVAL_POLICIES)}, got {eval_policy!r}"
        )

    anchors: list[AnchorEntry] = []
    excluded: list[ExcludedEntry] = []
    reasons_by_id: dict[str, list[str]] = {}
    candidates: list[Mapping[str, Any]] = []

    for entry in entries:
        entry_id = str(entry.get("id"))
        if entry.get("is_diagnostic_anchor"):
            anchors.append(AnchorEntry(id=entry_id, source_path=entry.get("source_path")))
            continue
        split = entry.get("split")
        if split not in (TRAIN, EVAL):
            excluded.append(
                ExcludedEntry(
                    id=entry_id,
                    reason=f"split={split!r} is not marked for train or eval",
                )
            )
            continue
        candidates.append(entry)

    survivors: list[Mapping[str, Any]] = []
    for entry in candidates:
        entry_id = str(entry.get("id"))
        issues = validate_entry(entry)
        if issues:
            reasons_by_id.setdefault(entry_id, []).extend(i.code for i in issues)
        else:
            survivors.append(entry)

    cross_issues = find_cross_entry_issues(survivors)
    conflicted_ids = set(cross_issues.keys())
    for entry_id, issues in cross_issues.items():
        reasons_by_id.setdefault(entry_id, []).extend(i.code for i in issues)
    survivors = [e for e in survivors if str(e.get("id")) not in conflicted_ids]

    final: list[Mapping[str, Any]] = []
    for entry in survivors:
        entry_id = str(entry.get("id"))
        split = entry["split"]
        export_path = _export_path(export_root, split, entry)
        if not _is_within_root(export_root, export_path):
            reasons_by_id.setdefault(entry_id, []).append("export_path_outside_root")
            continue
        if exists_fn(export_path):
            reasons_by_id.setdefault(entry_id, []).append(
                "output_path_would_overwrite_existing_file"
            )
            continue
        final.append(entry)

    train_entries = [e for e in final if e["split"] == TRAIN]
    eval_entries = [e for e in final if e["split"] == EVAL]
    train_source_groups = {e.get("source_group") for e in train_entries}

    def _plan_kwargs(e: Mapping[str, Any], split: str) -> dict[str, Any]:
        return dict(
            id=str(e.get("id")),
            export_path=_export_path(export_root, split, e),
            source_path=str(e.get("source_path")),
            source_sha256=str(e.get("source_sha256")),
            reviewed_clip_path=str(e.get("path")),
            reviewed_clip_sha256=str(e.get("sha256")),
            start_seconds=e.get("start_seconds"),
            end_seconds=e.get("end_seconds"),
            text=str(e.get("text")),
        )

    train_plans = [TrainEntryPlan(**_plan_kwargs(e, TRAIN)) for e in train_entries]

    eval_plans = []
    for e in eval_entries:
        # eval_kind always reflects the actual measured session relationship,
        # independent of eval_policy: the policy only decides whether a
        # within-session entry is allowed to count toward readiness, never
        # whether it gets relabelled as session-disjoint.
        kind = (
            WITHIN_SESSION_BLOCKED
            if e.get("source_group") in train_source_groups
            else SESSION_DISJOINT
        )
        eval_plans.append(
            EvalEntryPlan(**_plan_kwargs(e, EVAL), eval_kind=kind)
        )

    rejected = [
        RejectedEntry(id=entry_id, reasons=sorted(set(reasons)))
        for entry_id, reasons in reasons_by_id.items()
    ]

    blocking_reasons: list[str] = []
    if not dry_run_view and exists_fn(export_root):
        blocking_reasons.append(
            f"export root {export_root!r} already exists; refusing to plan into "
            "a pre-existing root outside an explicit dry-run view"
        )

    if not train_plans:
        blocking_reasons.append("train split is empty")

    if not eval_plans:
        blocking_reasons.append("eval split is empty")
    elif eval_policy == SESSION_DISJOINT:
        if not any(p.eval_kind == SESSION_DISJOINT for p in eval_plans):
            blocking_reasons.append(
                "eval split only contains within-session (blocked) entries; "
                "a session-disjoint eval source is required "
                "(or pass eval_policy='within_session_blocked' for an "
                "explicit bounded pilot)"
            )
    # eval_policy == WITHIN_SESSION_BLOCKED: any nonempty eval split
    # (session-disjoint or within-session) satisfies readiness; the
    # eval_kind label on each entry still tells the truth about which one.

    metadata_ready = not blocking_reasons

    return ExportPlan(
        train=train_plans,
        eval=eval_plans,
        diagnostic_anchors=anchors,
        excluded=excluded,
        rejected=rejected,
        eval_policy=eval_policy,
        metadata_ready=metadata_ready,
        blocking_reasons=blocking_reasons,
    )
