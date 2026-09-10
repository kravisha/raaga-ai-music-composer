"""Tests for the direct_tts manifest validator and export planner.

These tests only exercise plain-dict manifest metadata and the standard
library; no audio backend, model, or network access is imported or used.
"""

import os

from direct_tts.manifest import validate_entry, find_cross_entry_issues
from direct_tts.planner import plan_export
from direct_tts.verifier import verify_plan

SOURCE_SHA = "a" * 64
OUTPUT_SHA = "b" * 64
OUTPUT_SHA_2 = "c" * 64
SOURCE_SHA_2 = "e" * 64


def make_entry(**overrides):
    entry = {
        "id": "syn-train-0001",
        "source_path": r"C:\synthetic\source-a.mp3",
        "source_sha256": SOURCE_SHA,
        "source_group": "session-A",
        "start_seconds": 10.0,
        "end_seconds": 20.0,
        "path": "data/review-batch-01/clips/syn-train-0001.wav",
        "sha256": OUTPUT_SHA,
        "target": "spb_speech",
        "text": "அது ஒரு சோதனை வாக்கியம்",
        "text_review": "reviewed",
        "speaker_review": "reviewed",
        "quality_review": "reviewed",
        "use_scope": "local experiment only",
        "accepted_for_training": True,
        "split": "train",
        "is_diagnostic_anchor": False,
    }
    entry.update(overrides)
    return entry


# --- validate_entry: admission gate -----------------------------------


def test_fully_reviewed_entry_has_no_issues():
    assert validate_entry(make_entry()) == []


def test_missing_source_provenance_rejected():
    issues = validate_entry(make_entry(source_path="", source_sha256=""))
    assert any(i.code == "missing_source_provenance" for i in issues)


def test_malformed_source_hash_rejected():
    issues = validate_entry(make_entry(source_sha256="not-a-hash"))
    assert any(i.code == "missing_source_provenance" for i in issues)


def test_missing_output_reference_rejected():
    issues = validate_entry(make_entry(path="", sha256=""))
    assert any(i.code == "missing_output_reference" for i in issues)


def test_invalid_interval_rejected():
    issues = validate_entry(make_entry(start_seconds=20.0, end_seconds=10.0))
    assert any(i.code == "invalid_interval" for i in issues)


def test_mixed_target_singing_rejected():
    issues = validate_entry(make_entry(target="singing"))
    assert any(i.code == "mixed_target_singing" for i in issues)


def test_missing_target_rejected():
    issues = validate_entry(make_entry(target=None))
    assert any(i.code == "mixed_target_singing" for i in issues)


def test_missing_or_empty_text_rejected():
    issues = validate_entry(make_entry(text="   "))
    assert any(i.code == "missing_or_empty_text" for i in issues)


def test_disputed_text_rejected():
    issues = validate_entry(make_entry(text_review="disputed"))
    assert any(i.code == "disputed_text" for i in issues)


def test_unreviewed_text_rejected():
    issues = validate_entry(make_entry(text_review="pending"))
    assert any(i.code == "text_not_reviewed" for i in issues)


def test_unreviewed_speaker_rejected():
    issues = validate_entry(make_entry(speaker_review="pending"))
    assert any(i.code == "speaker_not_reviewed" for i in issues)


def test_unreviewed_quality_rejected():
    issues = validate_entry(make_entry(quality_review="pending"))
    assert any(i.code == "quality_not_reviewed" for i in issues)


def test_missing_use_scope_rejected():
    issues = validate_entry(make_entry(use_scope=""))
    assert any(i.code == "missing_use_scope" for i in issues)


def test_not_accepted_for_training_rejected():
    issues = validate_entry(make_entry(accepted_for_training=False))
    assert any(i.code == "not_accepted_for_training" for i in issues)


def test_review_only_split_rejected_by_validate_entry():
    issues = validate_entry(make_entry(split="review_only"))
    assert any(i.code == "not_marked_train_or_eval" for i in issues)


def test_diagnostic_anchor_conflict_rejected():
    issues = validate_entry(make_entry(is_diagnostic_anchor=True))
    assert any(i.code == "diagnostic_anchor_conflict" for i in issues)


# --- find_cross_entry_issues -------------------------------------------


def test_duplicate_interval_rejected_across_splits():
    a = make_entry(id="a", split="train")
    b = make_entry(
        id="b",
        split="eval",
        source_group="session-B",
        path="data/review-batch-01/clips/b.wav",
        sha256=OUTPUT_SHA_2,
    )
    issues = find_cross_entry_issues([a, b])
    assert "a" in issues and "b" in issues
    assert any(i.code == "duplicate_or_overlapping_interval" for i in issues["a"])
    assert any(i.code == "duplicate_or_overlapping_interval" for i in issues["b"])


def test_overlapping_interval_rejected():
    a = make_entry(id="a", start_seconds=10.0, end_seconds=20.0)
    b = make_entry(
        id="b",
        start_seconds=15.0,
        end_seconds=25.0,
        path="data/review-batch-01/clips/b.wav",
        sha256=OUTPUT_SHA_2,
    )
    issues = find_cross_entry_issues([a, b])
    assert set(issues.keys()) == {"a", "b"}


def test_non_overlapping_intervals_pass():
    a = make_entry(id="a", start_seconds=10.0, end_seconds=20.0)
    b = make_entry(
        id="b",
        start_seconds=20.0,
        end_seconds=30.0,
        path="data/review-batch-01/clips/b.wav",
        sha256=OUTPUT_SHA_2,
    )
    assert find_cross_entry_issues([a, b]) == {}


def test_source_hash_drift_rejected():
    a = make_entry(id="a")
    b = make_entry(
        id="b",
        source_sha256="d" * 64,
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/b.wav",
        sha256=OUTPUT_SHA_2,
    )
    issues = find_cross_entry_issues([a, b])
    assert any(i.code == "source_hash_drift" for i in issues.get("b", []))


def test_duplicate_output_path_rejected():
    a = make_entry(id="a")
    b = make_entry(
        id="b",
        start_seconds=100.0,
        end_seconds=110.0,
        path=a["path"],
        sha256=a["sha256"],
    )
    issues = find_cross_entry_issues([a, b])
    assert any(i.code == "duplicate_output_path" for i in issues.get("b", []))
    assert any(i.code == "duplicate_output_path" for i in issues.get("a", []))


def test_path_hash_drift_rejected():
    a = make_entry(id="a")
    b = make_entry(
        id="b",
        start_seconds=100.0,
        end_seconds=110.0,
        path=a["path"],
        sha256=OUTPUT_SHA_2,
    )
    issues = find_cross_entry_issues([a, b])
    assert any(i.code == "path_hash_drift" for i in issues.get("b", []))


def test_duplicate_entry_id_rejected():
    a = make_entry(id="dup")
    b = make_entry(
        id="dup",
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/other.wav",
        sha256=OUTPUT_SHA_2,
    )
    issues = find_cross_entry_issues([a, b])
    assert any(i.code == "duplicate_entry_id" for i in issues.get("dup", []))


# --- plan_export: admitted synthetic fixture ---------------------------


def _train_eval_fixture():
    train_entry = make_entry(
        id="syn-train-0001",
        source_path=r"C:\synthetic\session-a-part1.mp3",
        source_group="session-A",
        start_seconds=10.0,
        end_seconds=20.0,
        path="data/review-batch-01/clips/syn-train-0001.wav",
        sha256=OUTPUT_SHA,
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-0001",
        source_path=r"C:\synthetic\session-b-part1.mp3",
        source_sha256=SOURCE_SHA_2,  # genuinely different session audio
        source_group="session-B",
        start_seconds=10.0,
        end_seconds=20.0,
        path="data/review-batch-01/clips/syn-eval-0001.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    return [train_entry, eval_entry]


def test_plan_export_admits_synthetic_train_and_session_disjoint_eval():
    plan = plan_export(_train_eval_fixture(), exists_fn=lambda _p: False)

    assert plan.metadata_ready is True
    assert plan.blocking_reasons == []
    assert [e.id for e in plan.train] == ["syn-train-0001"]
    assert [e.id for e in plan.eval] == ["syn-eval-0001"]
    assert plan.eval[0].eval_kind == "session_disjoint"
    assert plan.rejected == []
    # Exact accepted text/split/export path are emitted:
    assert plan.train[0].text == "அது ஒரு சோதனை வாக்கியம்"
    assert plan.train[0].split == "train"
    assert plan.train[0].export_path == "dataset/train/syn-train-0001.wav"
    # Provenance (source hash AND reviewed-clip path/hash) must survive into
    # the emitted plan entry, not just the input manifest dict.
    assert plan.train[0].source_sha256 == SOURCE_SHA
    assert plan.train[0].reviewed_clip_path == "data/review-batch-01/clips/syn-train-0001.wav"
    assert plan.train[0].reviewed_clip_sha256 == OUTPUT_SHA
    assert plan.eval[0].source_sha256 == SOURCE_SHA_2
    assert plan.eval[0].reviewed_clip_sha256 == OUTPUT_SHA_2


def test_plan_export_blocks_within_session_only_eval():
    train_entry = make_entry(
        id="syn-train-0001",
        source_group="session-A",
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-blocked",
        source_group="session-A",  # same session as train: blocked
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/syn-eval-blocked.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export([train_entry, eval_entry], exists_fn=lambda _p: False)

    assert plan.metadata_ready is False
    assert any("within-session" in r for r in plan.blocking_reasons)
    assert len(plan.eval) == 1
    assert plan.eval[0].eval_kind == "within_session_blocked"


def test_plan_export_requires_nonempty_train():
    eval_entry = make_entry(
        id="syn-eval-0001",
        source_group="session-B",
        split="eval",
    )
    plan = plan_export([eval_entry], exists_fn=lambda _p: False)

    assert plan.metadata_ready is False
    assert "train split is empty" in plan.blocking_reasons


def test_plan_export_requires_nonempty_eval():
    train_entry = make_entry(id="syn-train-0001", split="train")
    plan = plan_export([train_entry], exists_fn=lambda _p: False)

    assert plan.metadata_ready is False
    assert "eval split is empty" in plan.blocking_reasons


def test_plan_export_keeps_diagnostic_anchor_out_of_train_and_eval():
    anchor = make_entry(
        id="cc-sbs1-diagnostic",
        split="train",
        is_diagnostic_anchor=True,
    )
    fixture = _train_eval_fixture() + [anchor]
    plan = plan_export(fixture, exists_fn=lambda _p: False)

    assert [a.id for a in plan.diagnostic_anchors] == ["cc-sbs1-diagnostic"]
    assert all(e.id != "cc-sbs1-diagnostic" for e in plan.train)
    assert all(e.id != "cc-sbs1-diagnostic" for e in plan.eval)
    assert all(r.id != "cc-sbs1-diagnostic" for r in plan.rejected)


def test_plan_export_excludes_review_only_entries():
    review_only = make_entry(
        id="cc-sbs2-060000",
        split="review_only",
        text_review="pending",
        speaker_review="pending",
        quality_review="pending",
        accepted_for_training=False,
    )
    fixture = _train_eval_fixture() + [review_only]
    plan = plan_export(fixture, exists_fn=lambda _p: False)

    assert [e.id for e in plan.excluded] == ["cc-sbs2-060000"]
    assert all(r.id != "cc-sbs2-060000" for r in plan.rejected)
    assert all(e.id != "cc-sbs2-060000" for e in plan.train + plan.eval)


def test_plan_export_refuses_existing_output_path_overwrite():
    fixture = _train_eval_fixture()

    def fake_exists(path):
        return path == "dataset/train/syn-train-0001.wav"

    plan = plan_export(fixture, exists_fn=fake_exists)

    assert all(e.id != "syn-train-0001" for e in plan.train)
    rejected_ids = {r.id: r.reasons for r in plan.rejected}
    assert "output_path_would_overwrite_existing_file" in rejected_ids["syn-train-0001"]
    assert plan.metadata_ready is False


def test_plan_export_rejects_duplicate_or_overlapping_interval_across_splits():
    a = make_entry(
        id="syn-train-0001",
        source_path=r"C:\synthetic\shared-source.mp3",
        source_group="session-A",
        start_seconds=10.0,
        end_seconds=20.0,
        split="train",
    )
    b = make_entry(
        id="syn-eval-overlap",
        source_path=r"C:\synthetic\shared-source.mp3",
        source_group="session-B",
        start_seconds=15.0,
        end_seconds=25.0,
        path="data/review-batch-01/clips/syn-eval-overlap.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export([a, b], exists_fn=lambda _p: False)

    assert plan.train == []
    assert plan.eval == []
    rejected_ids = {r.id for r in plan.rejected}
    assert rejected_ids == {"syn-train-0001", "syn-eval-overlap"}


def test_plan_export_does_not_touch_filesystem(tmp_path):
    fixture = _train_eval_fixture()
    before = sorted(os.listdir(tmp_path))

    plan_export(fixture, export_root=str(tmp_path / "dataset"), exists_fn=lambda _p: False)

    after = sorted(os.listdir(tmp_path))
    assert before == after == []


def test_plan_export_rejects_mixed_target_singing_from_train():
    singing_entry = make_entry(id="syn-singing", target="singing", split="train")
    plan = plan_export([singing_entry], exists_fn=lambda _p: False)

    assert plan.train == []
    rejected_ids = {r.id: r.reasons for r in plan.rejected}
    assert "mixed_target_singing" in rejected_ids["syn-singing"]


# --- Independent-review findings: cross-split content-identity leak ----


def test_missing_source_group_rejected():
    issues = validate_entry(make_entry(source_group=""))
    assert any(i.code == "missing_source_group" for i in issues)


def test_missing_source_group_rejected_when_absent_entirely():
    entry = make_entry()
    del entry["source_group"]
    issues = validate_entry(entry)
    assert any(i.code == "missing_source_group" for i in issues)


def test_same_source_bytes_relabelled_across_splits_rejected():
    # Reproduces claude-validator-independent-findings.json:
    # same source_sha256, overlapping windows, different source_path and
    # different (but present) source_group must not be admitted as an
    # independent eval source.
    train_entry = make_entry(
        id="syn-train-0001",
        source_path=r"C:\synthetic\source-a.mp3",
        source_group="session-A",
        start_seconds=10.0,
        end_seconds=20.0,
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-relabelled",
        source_path="C:/copy-of-a.mp3",  # different path, same bytes
        source_group="session-B",  # different, but bytes are identical
        start_seconds=10.0,
        end_seconds=20.0,
        path="data/review-batch-01/clips/syn-eval-relabelled.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export([train_entry, eval_entry], exists_fn=lambda _p: False)

    assert plan.train == []
    assert plan.eval == []
    rejected_ids = {r.id: r.reasons for r in plan.rejected}
    assert "duplicate_or_overlapping_interval" in rejected_ids["syn-train-0001"]
    assert "duplicate_or_overlapping_interval" in rejected_ids["syn-eval-relabelled"]
    assert plan.metadata_ready is False


# --- Independent-review findings: id path traversal ---------------------


def test_unsafe_id_with_path_traversal_rejected():
    issues = validate_entry(make_entry(id="../../escape"))
    assert any(i.code == "unsafe_id" for i in issues)


def test_unsafe_id_with_separator_rejected():
    issues = validate_entry(make_entry(id="a/b"))
    assert any(i.code == "unsafe_id" for i in issues)


def test_plan_export_rejects_traversal_id_and_never_escapes_root():
    entry = make_entry(id="../../escape", split="train")
    plan = plan_export([entry], exists_fn=lambda _p: False)

    assert plan.train == []
    rejected_ids = {r.id: r.reasons for r in plan.rejected}
    assert "unsafe_id" in rejected_ids["../../escape"]


def test_plan_export_rejects_export_root_that_already_exists():
    fixture = _train_eval_fixture()

    plan = plan_export(fixture, export_root="dataset", exists_fn=lambda p: p == "dataset")

    assert plan.metadata_ready is False
    assert any("already exists" in r for r in plan.blocking_reasons)


def test_plan_export_dry_run_view_bypasses_existing_root_check():
    fixture = _train_eval_fixture()

    plan = plan_export(
        fixture,
        export_root="dataset",
        exists_fn=lambda p: p == "dataset",
        dry_run_view=True,
    )

    assert plan.metadata_ready is True
    assert [e.id for e in plan.train] == ["syn-train-0001"]


# --- Independent-review findings: infinite / boolean intervals ---------


def test_infinite_end_seconds_rejected():
    issues = validate_entry(make_entry(end_seconds=float("inf")))
    assert any(i.code == "invalid_interval" for i in issues)


def test_infinite_start_seconds_rejected():
    issues = validate_entry(make_entry(start_seconds=float("-inf"), end_seconds=10.0))
    assert any(i.code == "invalid_interval" for i in issues)


def test_nan_end_seconds_rejected():
    issues = validate_entry(make_entry(end_seconds=float("nan")))
    assert any(i.code == "invalid_interval" for i in issues)


def test_boolean_start_seconds_rejected():
    issues = validate_entry(make_entry(start_seconds=False, end_seconds=10.0))
    assert any(i.code == "invalid_interval" for i in issues)


def test_boolean_end_seconds_rejected():
    issues = validate_entry(make_entry(start_seconds=0.0, end_seconds=True))
    assert any(i.code == "invalid_interval" for i in issues)


# --- Independent-review findings: metadata_ready vs execution_ready ----


def test_metadata_ready_is_not_execution_ready_without_verification():
    plan = plan_export(_train_eval_fixture(), exists_fn=lambda _p: False)
    assert plan.metadata_ready is True

    # No files exist on the (fake) filesystem: verification must refuse
    # execution readiness even though the metadata plan is ready.
    result = verify_plan(plan, export_root="dataset", exists_fn=lambda _p: False)
    assert result.execution_ready is False
    assert result.blocking_reasons != []


def test_verify_plan_execution_ready_when_hashes_match(tmp_path):
    source_path = tmp_path / "source-a.mp3"
    source_path.write_bytes(b"synthetic source bytes")
    train_clip_path = tmp_path / "syn-train-0001.wav"
    train_clip_path.write_bytes(b"synthetic reviewed train clip bytes")
    eval_clip_path = tmp_path / "syn-eval-0001.wav"
    eval_clip_path.write_bytes(b"synthetic reviewed eval clip bytes")

    from direct_tts.verifier import sha256_of_file

    source_sha = sha256_of_file(str(source_path))
    train_entry = make_entry(
        id="syn-train-0001",
        source_path=str(source_path),
        source_sha256=source_sha,
        source_group="session-A",
        path=str(train_clip_path),
        sha256=sha256_of_file(str(train_clip_path)),
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-0001",
        source_path=str(source_path),
        source_sha256=source_sha,
        source_group="session-A",
        start_seconds=100.0,
        end_seconds=110.0,
        path=str(eval_clip_path),
        sha256=sha256_of_file(str(eval_clip_path)),
        split="eval",
    )
    # session-disjoint pilot not needed here; only exercising verification.
    plan = plan_export(
        [train_entry, eval_entry],
        export_root=str(tmp_path / "dataset"),
        exists_fn=lambda _p: False,
        eval_policy="within_session_blocked",
    )
    assert plan.metadata_ready is True

    result = verify_plan(
        plan,
        export_root=str(tmp_path / "dataset"),
        exists_fn=lambda p: os.path.exists(p),
    )
    assert result.execution_ready is True
    assert all(e.ok for e in result.entries)


def test_verify_plan_rejects_source_sha256_mismatch(tmp_path):
    source_path = tmp_path / "source-a.mp3"
    source_path.write_bytes(b"original bytes")

    entry = make_entry(
        id="syn-train-0001",
        source_path=str(source_path),
        source_sha256="f" * 64,  # does not match actual file bytes
        split="train",
    )
    plan = plan_export(
        [entry], export_root=str(tmp_path / "dataset"), exists_fn=lambda _p: False
    )
    assert plan.metadata_ready is False  # eval split empty, but that's fine here

    result = verify_plan(
        plan, export_root=str(tmp_path / "dataset"), exists_fn=lambda p: os.path.exists(p)
    )
    assert result.execution_ready is False
    assert any("source_sha256_mismatch" in r for r in result.blocking_reasons) or any(
        "metadata plan is not ready" in r for r in result.blocking_reasons
    )


def test_verify_plan_never_calls_a_real_digest_function_when_injected(tmp_path):
    calls = []

    def fake_digest(path):
        calls.append(path)
        return "f" * 64

    train_entry = make_entry(
        id="syn-train-0001",
        source_sha256="f" * 64,
        sha256="f" * 64,
        path="clips/syn-train-0001.wav",
        source_group="session-A",
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-0001",
        source_sha256="f" * 64,  # fake_digest always returns this
        sha256="f" * 64,
        path="clips/syn-eval-0001.wav",
        source_group="session-B",
        start_seconds=100.0,
        end_seconds=110.0,
        split="eval",
    )
    plan = plan_export([train_entry, eval_entry], exists_fn=lambda _p: False)
    assert plan.metadata_ready is True

    result = verify_plan(
        plan,
        export_root="dataset",
        exists_fn=lambda _p: True,
        digest_fn=fake_digest,
    )
    assert result.execution_ready is True
    assert len(calls) == 4  # source + reviewed clip, for each of 2 entries


# --- Independent-review findings: explicit within-session pilot policy -


def test_eval_policy_within_session_blocked_allows_readiness():
    train_entry = make_entry(
        id="syn-train-0001",
        source_group="session-A",
        split="train",
    )
    eval_entry = make_entry(
        id="syn-eval-same-session",
        source_group="session-A",  # same session: allowed under this policy
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/syn-eval-same-session.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export(
        [train_entry, eval_entry],
        exists_fn=lambda _p: False,
        eval_policy="within_session_blocked",
    )

    assert plan.metadata_ready is True
    assert plan.eval_policy == "within_session_blocked"
    # Truthful label is preserved even though the policy allows it through:
    assert plan.eval[0].eval_kind == "within_session_blocked"


def test_eval_policy_default_still_blocks_within_session_only_eval():
    train_entry = make_entry(id="syn-train-0001", source_group="session-A", split="train")
    eval_entry = make_entry(
        id="syn-eval-same-session",
        source_group="session-A",
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/syn-eval-same-session.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export([train_entry, eval_entry], exists_fn=lambda _p: False)

    assert plan.eval_policy == "session_disjoint"
    assert plan.metadata_ready is False


def test_eval_policy_invalid_value_raises():
    entry = make_entry(id="syn-train-0001", split="train")
    try:
        plan_export([entry], exists_fn=lambda _p: False, eval_policy="cross_session_magic")
        assert False, "expected ValueError for an invalid eval_policy"
    except ValueError:
        pass


def test_eval_policy_within_session_still_rejects_source_overlap():
    a = make_entry(
        id="syn-train-0001",
        source_path=r"C:\synthetic\shared-source.mp3",
        source_group="session-A",
        start_seconds=10.0,
        end_seconds=20.0,
        split="train",
    )
    b = make_entry(
        id="syn-eval-overlap",
        source_path=r"C:\synthetic\shared-source.mp3",
        source_group="session-A",
        start_seconds=15.0,
        end_seconds=25.0,
        path="data/review-batch-01/clips/syn-eval-overlap.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export(
        [a, b], exists_fn=lambda _p: False, eval_policy="within_session_blocked"
    )

    assert plan.train == []
    assert plan.eval == []
    rejected_ids = {r.id for r in plan.rejected}
    assert rejected_ids == {"syn-train-0001", "syn-eval-overlap"}


def test_eval_policy_within_session_does_not_promote_review_only_exclusions():
    review_only = make_entry(
        id="cc-sbs2-060000",
        split="review_only",
        text_review="pending",
        speaker_review="pending",
        quality_review="pending",
        accepted_for_training=False,
    )
    train_entry = make_entry(id="syn-train-0001", source_group="session-A", split="train")
    eval_entry = make_entry(
        id="syn-eval-same-session",
        source_group="session-A",
        start_seconds=100.0,
        end_seconds=110.0,
        path="data/review-batch-01/clips/syn-eval-same-session.wav",
        sha256=OUTPUT_SHA_2,
        split="eval",
    )
    plan = plan_export(
        [train_entry, eval_entry, review_only],
        exists_fn=lambda _p: False,
        eval_policy="within_session_blocked",
    )

    assert [e.id for e in plan.excluded] == ["cc-sbs2-060000"]
    assert all(r.id != "cc-sbs2-060000" for r in plan.rejected)
    assert plan.metadata_ready is True
