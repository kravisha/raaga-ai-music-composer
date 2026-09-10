"""Filesystem-backed verification that a metadata-approved export plan is
actually safe to execute.

``planner.plan_export`` only reasons about manifest dict fields; it never
opens a file. This module is the explicit, separate step that does: for
every entry in an already metadata-ready plan, it confirms the source and
reviewed-clip files exist, that their bytes still match the recorded
SHA-256, and that the resolved export path stays inside the requested
root. It still never decodes audio or loads a model -- the digest function
is injectable so tests exercise this with plain temporary files, not real
audio.

``VerificationResult.execution_ready`` is the only signal in this package
that means "safe to actually copy/export files"; ``ExportPlan.metadata_ready``
alone is not sufficient, by design.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
from dataclasses import dataclass, field
from typing import Callable, Sequence

from direct_tts.planner import EvalEntryPlan, ExportPlan, TrainEntryPlan

_CHUNK_SIZE = 1024 * 1024


def sha256_of_file(path: str) -> str:
    """Stdlib-only streaming SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EntryVerification:
    id: str
    ok: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationResult:
    execution_ready: bool
    blocking_reasons: list[str] = field(default_factory=list)
    entries: list[EntryVerification] = field(default_factory=list)


def _is_within_root(export_root: str, path: str) -> bool:
    root_norm = posixpath.normpath(export_root)
    path_norm = posixpath.normpath(path)
    return path_norm == root_norm or path_norm.startswith(root_norm + "/")


def _verify_one(
    entry: TrainEntryPlan | EvalEntryPlan,
    export_root: str,
    exists_fn: Callable[[str], bool],
    digest_fn: Callable[[str], str],
) -> EntryVerification:
    reasons: list[str] = []

    if not _is_within_root(export_root, entry.export_path):
        reasons.append("export_path_outside_root")

    if not exists_fn(entry.source_path):
        reasons.append("source_file_missing")
    else:
        actual = digest_fn(entry.source_path)
        if actual.lower() != entry.source_sha256.lower():
            reasons.append("source_sha256_mismatch")

    if not exists_fn(entry.reviewed_clip_path):
        reasons.append("reviewed_clip_missing")
    else:
        actual_clip = digest_fn(entry.reviewed_clip_path)
        if actual_clip.lower() != entry.reviewed_clip_sha256.lower():
            reasons.append("reviewed_clip_sha256_mismatch")

    return EntryVerification(id=entry.id, ok=not reasons, reasons=reasons)


def verify_plan(
    plan: ExportPlan,
    export_root: str,
    exists_fn: Callable[[str], bool] = os.path.exists,
    digest_fn: Callable[[str], str] = sha256_of_file,
) -> VerificationResult:
    """Run the filesystem-backed checks a metadata plan cannot perform.

    Never mutates the filesystem; only reads bytes to compute digests via
    ``digest_fn`` (or checks existence via ``exists_fn``). Both are
    injectable so tests never need a real audio file or a real dataset
    root.
    """
    blocking: list[str] = []
    if not plan.metadata_ready:
        blocking.append(
            "metadata plan is not ready: " + "; ".join(plan.blocking_reasons)
        )

    entry_results: list[EntryVerification] = []
    entries: Sequence[TrainEntryPlan | EvalEntryPlan] = list(plan.train) + list(plan.eval)
    for entry in entries:
        result = _verify_one(entry, export_root, exists_fn, digest_fn)
        entry_results.append(result)
        if not result.ok:
            blocking.append(f"{result.id}: " + ", ".join(result.reasons))

    execution_ready = plan.metadata_ready and not blocking

    return VerificationResult(
        execution_ready=execution_ready,
        blocking_reasons=blocking,
        entries=entry_results,
    )
