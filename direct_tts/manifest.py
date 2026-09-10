"""Per-entry and cross-entry manifest validation.

Operates purely on plain dicts describing manifest entries (as loaded from
JSON). No file I/O, no audio decoding: this module only reasons about the
metadata a manifest records, so it can be unit tested without touching any
audio backend.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Only this exact target label admits an entry into speech training. Anything
# else (singing, interviewer, krish_voice, music, missing) is excluded and,
# if it was nonetheless flagged for train/eval, rejected as a mixed target.
ADMITTED_TARGET = "spb_speech"

REVIEWED = "reviewed"
DISPUTED = "disputed"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# A filename stem only: no path separators, no leading dot (so a lone ".."
# or a hidden-file-style id can never appear), safe to join under an export
# root without escaping it. This is the admission-time half of path-traversal
# defense; planner.py additionally verifies the joined path stays under the
# requested root before it is ever handed back to a caller.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

TRAIN = "train"
EVAL = "eval"
REVIEW_ONLY = "review_only"
_TRAIN_OR_EVAL = frozenset({TRAIN, EVAL})


@dataclass(frozen=True)
class ValidationIssue:
    entry_id: str
    code: str
    message: str


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def validate_entry(entry: Mapping[str, Any]) -> list[ValidationIssue]:
    """Validate a single manifest entry against the admission contract.

    Returns an empty list if the entry is eligible to be placed into a
    train or eval split. Entries whose declared ``split`` is not ``train``
    or ``eval`` (e.g. ``review_only``, pending, or absent) are not admission
    candidates and are not evaluated here; callers should route those to an
    "excluded" bucket instead of calling this function on them.
    """
    entry_id = str(entry.get("id") or "<missing-id>")
    issues: list[ValidationIssue] = []

    def reject(code: str, message: str) -> None:
        issues.append(ValidationIssue(entry_id, code, message))

    if not _is_nonempty_str(entry.get("id")):
        reject("missing_id", "entry has no nonempty id")
    elif not _SAFE_ID_RE.match(entry["id"]):
        reject(
            "unsafe_id",
            f"id {entry['id']!r} is not a safe filename stem (no path "
            "separators, must start with a letter or digit)",
        )

    if not _is_nonempty_str(entry.get("source_path")) or not _is_sha256(
        entry.get("source_sha256")
    ):
        reject(
            "missing_source_provenance",
            "source_path and a 64-hex-char source_sha256 are required",
        )

    if not _is_nonempty_str(entry.get("source_group")):
        reject(
            "missing_source_group",
            "source_group is required and must be nonempty; a missing "
            "group is never treated as independent of other entries",
        )

    if not _is_nonempty_str(entry.get("path")) or not _is_sha256(entry.get("sha256")):
        reject(
            "missing_output_reference",
            "path and a 64-hex-char sha256 for the reviewed clip are required",
        )

    start = entry.get("start_seconds")
    end = entry.get("end_seconds")

    def _is_finite_number(value: Any) -> bool:
        # bool is a subclass of int, so it must be excluded explicitly, and
        # inf/nan are technically floats but never a valid clip boundary.
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if not _is_finite_number(start) or not _is_finite_number(end) or not (
        end > start >= 0
    ):
        reject(
            "invalid_interval",
            "start_seconds/end_seconds must be finite, non-boolean numbers "
            f"satisfying end > start >= 0, got {start!r}/{end!r}",
        )

    if entry.get("is_diagnostic_anchor"):
        reject(
            "diagnostic_anchor_conflict",
            "a diagnostic/development anchor may never be placed into train or eval",
        )

    target = entry.get("target")
    if target != ADMITTED_TARGET:
        reject(
            "mixed_target_singing",
            f"target={target!r} is not the admitted training target "
            f"{ADMITTED_TARGET!r}; singing/interviewer/other speakers may "
            "not be pooled into speech training",
        )

    text = entry.get("text")
    if not _is_nonempty_str(text):
        reject("missing_or_empty_text", "reviewed text is missing or empty")

    text_review = entry.get("text_review")
    if text_review == DISPUTED:
        reject("disputed_text", "text_review is disputed")
    elif text_review != REVIEWED:
        reject(
            "text_not_reviewed",
            f"text_review={text_review!r}, expected {REVIEWED!r}",
        )

    speaker_review = entry.get("speaker_review")
    if speaker_review != REVIEWED:
        reject(
            "speaker_not_reviewed",
            f"speaker_review={speaker_review!r}, expected {REVIEWED!r}",
        )

    quality_review = entry.get("quality_review")
    if quality_review != REVIEWED:
        reject(
            "quality_not_reviewed",
            f"quality_review={quality_review!r}, expected {REVIEWED!r}",
        )

    if not _is_nonempty_str(entry.get("use_scope")):
        reject("missing_use_scope", "recorded use_scope is missing or empty")

    if entry.get("accepted_for_training") is not True:
        reject(
            "not_accepted_for_training",
            "accepted_for_training must be explicitly true",
        )

    split = entry.get("split")
    if split not in _TRAIN_OR_EVAL:
        reject(
            "not_marked_train_or_eval",
            f"split={split!r} is not 'train' or 'eval'",
        )

    return issues


def _intervals_conflict(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    # Identity is content, not the recorded path/name: two entries can carry
    # different source_path strings (a copy, a different mount point, a
    # relabelled session) and still be bytes-identical source audio. Any
    # such pair sharing an overlapping/duplicate window must be rejected
    # regardless of source_path or source_group, so this keys on
    # source_sha256 rather than source_path. Callers are expected to only
    # pass entries that already passed validate_entry, so both hashes are
    # guaranteed to be well-formed 64-hex-char strings here.
    a_hash, b_hash = a.get("source_sha256"), b.get("source_sha256")
    if not (_is_sha256(a_hash) and _is_sha256(b_hash)):
        return False
    if str(a_hash).lower() != str(b_hash).lower():
        return False
    a_start, a_end = a.get("start_seconds"), a.get("end_seconds")
    b_start, b_end = b.get("start_seconds"), b.get("end_seconds")
    if None in (a_start, a_end, b_start, b_end):
        return False
    if a_start == b_start and a_end == b_end:
        return True  # exact duplicate window
    return a_start < b_end and b_start < a_end  # overlap


def find_cross_entry_issues(
    entries: Sequence[Mapping[str, Any]]
) -> dict[str, list[ValidationIssue]]:
    """Find conflicts among a set of entries that are each individually
    admission-eligible (callers should pass only entries that already
    passed :func:`validate_entry`).

    Returns a mapping of entry_id -> list of ValidationIssue for every
    entry involved in a conflict. An entry with no issues is not present
    in the returned mapping.
    """
    issues: dict[str, list[ValidationIssue]] = {}

    def add(entry_id: str, code: str, message: str) -> None:
        issues.setdefault(entry_id, []).append(ValidationIssue(entry_id, code, message))

    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for e in entries:
        by_id.setdefault(str(e.get("id")), []).append(e)
    for entry_id, group in by_id.items():
        if len(group) > 1:
            for e in group:
                add(
                    entry_id,
                    "duplicate_entry_id",
                    f"id {entry_id!r} appears {len(group)} times in the manifest",
                )

    by_source_path: dict[str, str] = {}
    for e in entries:
        sp = e.get("source_path")
        sh = e.get("source_sha256")
        entry_id = str(e.get("id"))
        if sp is None:
            continue
        if sp in by_source_path and by_source_path[sp] != sh:
            add(
                entry_id,
                "source_hash_drift",
                f"source_path {sp!r} recorded with conflicting source_sha256 values",
            )
        else:
            by_source_path[sp] = sh

    by_output_path: dict[str, tuple[str, Any]] = {}
    for e in entries:
        op = e.get("path")
        oh = e.get("sha256")
        entry_id = str(e.get("id"))
        if op is None:
            continue
        if op in by_output_path:
            prev_id, prev_hash = by_output_path[op]
            if prev_hash != oh:
                add(
                    entry_id,
                    "path_hash_drift",
                    f"output path {op!r} recorded with conflicting sha256 values "
                    f"(also used by {prev_id!r})",
                )
            else:
                add(
                    entry_id,
                    "duplicate_output_path",
                    f"output path {op!r} is reused by entry {prev_id!r}",
                )
                add(
                    prev_id,
                    "duplicate_output_path",
                    f"output path {op!r} is reused by entry {entry_id!r}",
                )
        else:
            by_output_path[op] = (entry_id, oh)

    n = len(entries)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = entries[i], entries[j]
            if _intervals_conflict(a, b):
                a_id, b_id = str(a.get("id")), str(b.get("id"))
                cross_split = a.get("split") != b.get("split")
                scope = "across splits" if cross_split else "within the same split"
                add(
                    a_id,
                    "duplicate_or_overlapping_interval",
                    f"source window overlaps/duplicates entry {b_id!r} ({scope})",
                )
                add(
                    b_id,
                    "duplicate_or_overlapping_interval",
                    f"source window overlaps/duplicates entry {a_id!r} ({scope})",
                )

    return issues
