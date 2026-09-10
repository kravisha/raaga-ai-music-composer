"""Dry-run dataset export planning.

Given a list of manifest entries (plain dicts, as loaded from JSON), decide
which ones are admitted into a train split and which into an eval split,
refusing on any contract violation. This module never opens an audio file,
never loads a model and never starts training: it only computes an export
plan (which reviewed clip would go where, under what new export path).
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


@dataclass(frozen=True)
class TrainEntryPlan:
    id: str
    export_path: str
    source_path: str
    start_seconds: float
    end_seconds: float
    text: str
    split: str = TRAIN


@dataclass(frozen=True)
class EvalEntryPlan:
    id: str
    export_path: str
    source_path: str
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
    ready: bool = False
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


def plan_export(
    entries: Sequence[Mapping[str, Any]],
    export_root: str = DEFAULT_EXPORT_ROOT,
    exists_fn: Callable[[str], bool] = os.path.exists,
) -> ExportPlan:
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
        if exists_fn(export_path):
            reasons_by_id.setdefault(entry_id, []).append(
                "output_path_would_overwrite_existing_file"
            )
            continue
        final.append(entry)

    train_entries = [e for e in final if e["split"] == TRAIN]
    eval_entries = [e for e in final if e["split"] == EVAL]
    train_source_groups = {e.get("source_group") for e in train_entries}

    train_plans = [
        TrainEntryPlan(
            id=str(e.get("id")),
            export_path=_export_path(export_root, TRAIN, e),
            source_path=str(e.get("source_path")),
            start_seconds=e.get("start_seconds"),
            end_seconds=e.get("end_seconds"),
            text=str(e.get("text")),
        )
        for e in train_entries
    ]

    eval_plans = []
    for e in eval_entries:
        kind = (
            "within_session_blocked"
            if e.get("source_group") in train_source_groups
            else "session_disjoint"
        )
        eval_plans.append(
            EvalEntryPlan(
                id=str(e.get("id")),
                export_path=_export_path(export_root, EVAL, e),
                source_path=str(e.get("source_path")),
                start_seconds=e.get("start_seconds"),
                end_seconds=e.get("end_seconds"),
                text=str(e.get("text")),
                eval_kind=kind,
            )
        )

    rejected = [
        RejectedEntry(id=entry_id, reasons=sorted(set(reasons)))
        for entry_id, reasons in reasons_by_id.items()
    ]

    blocking_reasons: list[str] = []
    if not train_plans:
        blocking_reasons.append("train split is empty")
    session_disjoint_eval = [p for p in eval_plans if p.eval_kind == "session_disjoint"]
    if not eval_plans:
        blocking_reasons.append("eval split is empty")
    elif not session_disjoint_eval:
        blocking_reasons.append(
            "eval split only contains within-session (blocked) entries; "
            "a session-disjoint eval source is required"
        )

    ready = not blocking_reasons

    return ExportPlan(
        train=train_plans,
        eval=eval_plans,
        diagnostic_anchors=anchors,
        excluded=excluded,
        rejected=rejected,
        ready=ready,
        blocking_reasons=blocking_reasons,
    )
