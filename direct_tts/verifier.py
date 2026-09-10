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
from dataclasses import dataclass, field
from typing import Callable, Sequence

from direct_tts._paths import is_within_root
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
    """Delegates to :func:`direct_tts._paths.is_within_root`, which resolves
    both paths against the real filesystem instead of doing purely lexical,
    forward-slash-only normalization (see that module's docstring)."""
    return is_within_root(export_root, path)


def _digest_or_block(
    path: str, digest_fn: Callable[[str], str], failure_reason: str, reasons: list[str]
) -> str | None:
    """Read/hash ``path``, returning the digest, or append ``failure_reason``
    and return ``None`` on any read error.

    A hash/read failure (permission error, race where the file vanished
    after the existence check, transient I/O error, ...) must never be
    allowed to raise out of verification and must never be silently
    treated as a pass; it always means "not verified", i.e. blocked.
    """
    try:
        return digest_fn(path)
    except OSError:
        reasons.append(failure_reason)
        return None


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
        actual = _digest_or_block(
            entry.source_path, digest_fn, "source_read_failed", reasons
        )
        if actual is not None and actual.lower() != entry.source_sha256.lower():
            reasons.append("source_sha256_mismatch")

    if not exists_fn(entry.reviewed_clip_path):
        reasons.append("reviewed_clip_missing")
    else:
        actual_clip = _digest_or_block(
            entry.reviewed_clip_path, digest_fn, "reviewed_clip_read_failed", reasons
        )
        if (
            actual_clip is not None
            and actual_clip.lower() != entry.reviewed_clip_sha256.lower()
        ):
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

    This is a read-only check, not a race-free guarantee: it does not by
    itself prevent a concurrent writer from creating ``export_root`` (or
    changing source/clip bytes) immediately after this function returns.
    An actual exporter must still create its output exclusively (e.g. an
    exclusive-create mkdir/open) to close that window; this verifier only
    catches a stale/prior state, not a race with a concurrent writer.
    """
    blocking: list[str] = []
    if not plan.metadata_ready:
        blocking.append(
            "metadata plan is not ready: " + "; ".join(plan.blocking_reasons)
        )

    if plan.preview_only:
        # A dry-run/preview plan (plan_export(..., dry_run_view=True)) may
        # report metadata_ready=True so a caller can see what *would* be
        # planned, but it explicitly skipped the "export root must not
        # already exist" check to allow previewing against a root that may
        # already hold prior output. That preview status must never be
        # forgotten by the time we ask "is this safe to execute": refuse
        # unconditionally, independent of what the filesystem looks like
        # right now. A caller that genuinely wants to execute must request
        # a fresh, non-preview plan_export() once the root truly does not
        # exist yet.
        blocking.append(
            "plan was produced as a dry-run/preview view (dry_run_view=True); "
            "a preview plan is never execution-ready -- request a fresh "
            "non-preview plan_export() once the export root does not exist"
        )

    if exists_fn(export_root):
        # Always rechecked here, regardless of preview_only and regardless
        # of what plan_export saw at planning time: the root may have been
        # created by something else in the interim between planning and
        # this verification call.
        blocking.append(
            f"export root {export_root!r} already exists at verification "
            "time; refusing to treat the plan as execution-ready"
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
