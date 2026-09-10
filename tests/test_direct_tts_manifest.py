"""Tests for the direct_tts manifest validator and export planner.

These tests only exercise plain-dict manifest metadata and the standard
library; no audio backend, model, or network access is imported or used.
"""

import os

from direct_tts.manifest import validate_entry, find_cross_entry_issues
from direct_tts.planner import plan_export

SOURCE_SHA = "a" * 64
OUTPUT_SHA = "b" * 64
OUTPUT_SHA_2 = "c" * 64


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

    assert plan.ready is True
    assert plan.blocking_reasons == []
    assert [e.id for e in plan.train] == ["syn-train-0001"]
    assert [e.id for e in plan.eval] == ["syn-eval-0001"]
    assert plan.eval[0].eval_kind == "session_disjoint"
    assert plan.rejected == []
    # Exact accepted text/split/export path are emitted:
    assert plan.train[0].text == "அது ஒரு சோதனை வாக்கியம்"
    assert plan.train[0].split == "train"
    assert plan.train[0].export_path == "dataset/train/syn-train-0001.wav"


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

    assert plan.ready is False
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

    assert plan.ready is False
    assert "train split is empty" in plan.blocking_reasons


def test_plan_export_requires_nonempty_eval():
    train_entry = make_entry(id="syn-train-0001", split="train")
    plan = plan_export([train_entry], exists_fn=lambda _p: False)

    assert plan.ready is False
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
    assert plan.ready is False


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
