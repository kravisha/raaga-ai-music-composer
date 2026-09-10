"""Focused post-hoc checks on the collected-speech candidate dataset produced by
prepare_spb_collected_dataset.py. Pure validation of already-written artifacts; no
GPU/ASR/network/training work runs here.
"""
import hashlib
import json
from pathlib import Path

OUT = Path("C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910/data/collected-speech-candidates-01")


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load():
    return json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))


def test_output_files_exist():
    assert (OUT / "candidates.json").exists()
    assert (OUT / "REPORT.md").exists()
    assert (OUT / "status.json").exists()
    assert (OUT / "asr-locator-evidence.json").exists()


def test_schema_keys_present():
    d = _load()
    for key in ("status", "source_records", "coverage", "candidates", "exclusions", "gpu_released"):
        assert key in d
    assert d["gpu_released"] is True
    required = {"id", "path", "sha256", "text", "text_file", "text_sha256", "raw_text",
                "raw_text_origin_path", "raw_text_origin_sha256", "pronunciation_mappings",
                "source_path", "source_sha256", "source_group", "source_start_seconds",
                "source_end_seconds", "duration_seconds", "sample_rate", "channels",
                "parent_review_ids", "text_status", "alignment_evidence", "quality_evidence",
                "proposed_split", "uncertainty"}
    for c in d["candidates"]:
        missing = required - set(c.keys())
        assert not missing, f"{c['id']} missing {missing}"


def test_candidate_files_hash_and_duration_bounds():
    d = _load()
    for c in d["candidates"]:
        assert _sha256(c["path"]) == c["sha256"], c["id"]
        assert _sha256(c["text_file"]) == c["text_sha256"], c["id"]
        assert c["sample_rate"] == 16000
        assert c["channels"] == 1
        assert 2.0 <= c["duration_seconds"] <= 20.0, (c["id"], c["duration_seconds"])
        assert c["text_status"] in ("previously_human_approved", "source_checked_user_supplied")
        assert c["proposed_split"] in ("train", "holdout")


def test_text_status_matches_old_ledger():
    d = _load()
    old_ids = {"cc-sbs2-060000-body", "spb1-first-reviewed-105830", "spb1-reviewed-002-core", "spb1-reviewed-006-core"}
    for c in d["candidates"]:
        if c["id"] in old_ids:
            assert c["text_status"] == "previously_human_approved", c["id"]
        else:
            assert c["text_status"] == "source_checked_user_supplied", c["id"]


def test_no_duplicate_or_overlapping_source1_intervals():
    d = _load()
    intervals = sorted(
        (c["source_start_seconds"], c["source_end_seconds"], c["id"])
        for c in d["candidates"]
        if c["source_path"] == d["source_records"]["source1"]["path"]
    )
    for i in range(len(intervals) - 1):
        lo1, hi1, id1 = intervals[i]
        lo2, hi2, id2 = intervals[i + 1]
        assert hi1 <= lo2 + 1e-9, f"overlap between {id1} and {id2}"


def test_withdrawn_008_excluded():
    d = _load()
    for c in d["candidates"]:
        assert "008" not in c["id"], c["id"]
        assert "spb1-reviewed-008-core" != c["id"]


def test_holdout_gap_and_neighbours():
    d = _load()
    source1_path = d["source_records"]["source1"]["path"]
    train = [c for c in d["candidates"] if c["proposed_split"] == "train" and c["source_path"] == source1_path]
    holdouts = [c for c in d["candidates"] if c["proposed_split"] == "holdout"]
    assert len(holdouts) == 1
    h = holdouts[0]
    for t in train:
        gap = max(h["source_start_seconds"] - t["source_end_seconds"],
                   t["source_start_seconds"] - h["source_end_seconds"])
        assert gap >= 2.0, (t["id"], gap)


def test_coverage_accounting_closes():
    d = _load()
    assert d["coverage"]["source1_accounting_closes"] is True
    total = d["coverage"]["source1_candidate_seconds"] + d["coverage"]["source1_exclusion_seconds"]
    assert abs(total - d["coverage"]["source1_total_seconds"]) < 1e-6


def test_exclusions_are_gapless_partition_with_candidates():
    d = _load()
    source1_path = d["source_records"]["source1"]["path"]
    spans = [(c["source_start_seconds"], c["source_end_seconds"]) for c in d["candidates"] if c["source_path"] == source1_path]
    spans += [tuple(e["interval_source_seconds"]) for e in d["exclusions"]]
    spans.sort()
    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - d["coverage"]["source1_total_seconds"]) < 1e-6
    for i in range(len(spans) - 1):
        assert abs(spans[i][1] - spans[i + 1][0]) < 1e-6, (spans[i], spans[i + 1])


def test_four_old_pairs_preserved_and_unmodified():
    d = _load()
    old_shas = {
        "cc-sbs2-060000-body": "ab60cc1880aac0e6b23415ffd732fc8747601c284d1555a3c92ed6009d016af5",
        "spb1-first-reviewed-105830": "98a6be9c9b7dea214aa369487fa174fc94f4b3230cc41492787a10c98d7a19da",
        "spb1-reviewed-002-core": "730ba60f1d48f9a9cb25c292482d910d3c7f5a57c5bfca0d68de88141e60b930",
        "spb1-reviewed-006-core": "b1de0e457b95f8aa40430face7918843c2d26f62d39faf9b16ff98892efcc4d3",
    }
    found = {c["id"]: c["sha256"] for c in d["candidates"] if c["id"] in old_shas}
    assert found == old_shas


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
