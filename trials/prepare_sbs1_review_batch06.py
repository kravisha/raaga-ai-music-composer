"""Prepare the next two ~20s SBS1 review units (CC-SPB1-022..023 / GLOBAL21-22).

Single decode of the immutable source1 MP3 into float32 mono, then two
contiguous crops from core 329.050s onward (the end of raw CC-SPB1-021,
GLOBAL20, covered by the prior global20 review) with no gain/filter/speed/
pitch changes. Each unit is prepared directly as one continuous ~20s clip.
Core unit boundaries are chosen near a local short-frame RMS-energy minimum
inside an 18-22s search window after the previous core end -- this is an
energy heuristic for "near a quiet pause", not a certified word/speaker
boundary. These are human-review units, not admitted training segments
(training_admitted stays false).

This is an adapted copy of `prepare_sbs1_review_batch05.py` (this worktree)
with new constants for batch-06 (2 units, ~20s each, 18-22s search window);
that source script is not run or edited.

Output (exclusively owned by Claude Code in this assignment):
  data/review-sbs1-claude-batch-06/audio/CC-SPB1-0XX.wav
  data/review-sbs1-claude-batch-06/batch.json
  data/review-sbs1-claude-batch-06/READY.json
  data/review-sbs1-claude-batch-06/NOTES.md (written separately)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SOURCE = Path(r"C:\Users\Krish\Documents\Sound Recordings\SPB-interview-part-1.mp3")
SOURCE_SHA256_EXPECTED = "7429c56bc9fcf5e2c69942e4387f8448b0bd5ed99de67638e90de104c96740f7"
SOURCE_DURATION_EXPECTED = 829.632

COMMON_ROOT = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-SPB-Direct-TTS-20260910")
OUT_DIR = COMMON_ROOT / "data" / "review-sbs1-claude-batch-06"
AUDIO_DIR = OUT_DIR / "audio"

BATCH_ID = "review-sbs1-claude-batch-06"
OWNER = "claude-code"
WORKTREE = "raaga-wt-claude-spb-source1-batch2"
BRANCH = "claude/spb-source1-batch2-20260910"

PREVIOUS_UNIT = {
    "id": "CC-SPB1-021",
    "global_label": "GLOBAL20",
    "audio": str(
        COMMON_ROOT / "data" / "review-sbs1-claude-batch-05" / "audio" / "CC-SPB1-021.wav"
    ),
    "sha256": "bd4fbf30eb5f9e2e04b58570351294b5ef9304a71f98f53919f1364201ee9d28",
    "core_start_seconds": 309.305,
    "core_end_seconds": 329.050,
}

CORE_START_SECONDS = PREVIOUS_UNIT["core_end_seconds"]
NUM_UNITS = 2
ID_START = 22
GLOBAL_LABEL_START = 21
TARGET_UNIT_SECONDS = 20.0
MIN_UNIT_SECONDS = 18.0
MAX_UNIT_SECONDS = 22.0
PRE_ROLL_SECONDS = 0.30
PAUSE_FRAME_SECONDS = 0.030
PAUSE_HOP_SECONDS = 0.010


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_array(samples: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
    return h.hexdigest()


def find_quiet_boundary(samples: np.ndarray, sr: int, search_start_s: float, search_end_s: float):
    """Return (time_seconds, rms) of the quietest short frame in [search_start_s, search_end_s].

    Energy-only heuristic for "near a natural quiet pause". Not a word or
    speaker boundary by itself.
    """
    frame_len = max(1, int(round(PAUSE_FRAME_SECONDS * sr)))
    hop = max(1, int(round(PAUSE_HOP_SECONDS * sr)))
    start_sample = int(round(search_start_s * sr))
    end_sample = min(int(round(search_end_s * sr)), len(samples) - frame_len)
    if end_sample <= start_sample:
        clamped = max(0, min(len(samples) - 1, int(round(search_end_s * sr))))
        return clamped / sr, None

    best_center_sample = None
    best_rms = None
    pos = start_sample
    while pos <= end_sample:
        frame = samples[pos:pos + frame_len]
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_center_sample = pos + frame_len // 2
        pos += hop

    return best_center_sample / sr, best_rms


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_ready(entries, status):
    ready = {
        "batch_id": BATCH_ID,
        "owner": OWNER,
        "worktree": WORKTREE,
        "branch": BRANCH,
        "status": status,
        "first_ready_path": str((AUDIO_DIR / f"CC-SPB1-{ID_START:03d}.wav")) if entries else None,
        "batch_manifest_path": str(OUT_DIR / "batch.json"),
        "ordered_ids_ready": [e["id"] for e in entries],
        "global_labels_ready": [e["global_label"] for e in entries],
        "last_updated_utc": utc_now_iso(),
        "note": (
            "Human-review units only; not admitted training segments. "
            "Codex presents ONE AT A TIME for Krish's dictation/correction."
        ),
    }
    (OUT_DIR / "READY.json").write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    # Review audio can already carry human corrections. Refuse a repeat run
    # before decoding or writing anything, rather than overwriting a batch.
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    AUDIO_DIR.mkdir(exist_ok=False)

    source_sha_before = sha256_of_file(SOURCE)
    assert source_sha_before == SOURCE_SHA256_EXPECTED, (
        f"source hash mismatch before decode: {source_sha_before}"
    )

    samples, sr = sf.read(str(SOURCE), dtype="float32", always_2d=False)
    assert samples.ndim == 1, f"expected mono, got shape {samples.shape}"
    assert sr == 48000, f"expected 48kHz, got {sr}"
    source_duration = len(samples) / sr
    assert abs(source_duration - SOURCE_DURATION_EXPECTED) < 0.01, (
        f"expected duration {SOURCE_DURATION_EXPECTED}, got {source_duration}"
    )

    source_sha_after_decode = sha256_of_file(SOURCE)
    assert source_sha_after_decode == SOURCE_SHA256_EXPECTED, "source mutated during decode"

    assert abs(CORE_START_SECONDS - 329.050) < 1e-9, (
        f"expected first core to start EXACT 329.050, got {CORE_START_SECONDS}"
    )

    # Contiguous core boundaries, each unit's core starts exactly where the previous core ended.
    core_bounds = [CORE_START_SECONDS]
    boundary_info = []
    for _ in range(NUM_UNITS):
        start = core_bounds[-1]
        search_start = start + MIN_UNIT_SECONDS
        search_end = min(start + MAX_UNIT_SECONDS, source_duration)
        end, rms = find_quiet_boundary(samples, sr, search_start, search_end)
        core_bounds.append(end)
        boundary_info.append({
            "search_window_seconds": [round(search_start, 6), round(search_end, 6)],
            "boundary_rms": rms,
            "target_unit_seconds": TARGET_UNIT_SECONDS,
        })

    entries = []
    for i in range(NUM_UNITS):
        unit_id = f"CC-SPB1-{ID_START + i:03d}"
        global_label = f"GLOBAL{GLOBAL_LABEL_START + i}"
        core_start = core_bounds[i]
        core_end = core_bounds[i + 1]

        pre_roll = min(PRE_ROLL_SECONDS, core_start)  # never crop before source start
        audio_start = core_start - pre_roll
        audio_end = core_end

        core_start_sample = int(round(core_start * sr))
        core_end_sample = int(round(core_end * sr))
        audio_start_sample = int(round(audio_start * sr))
        audio_end_sample = int(round(audio_end * sr))

        crop = samples[audio_start_sample:audio_end_sample]
        source_slice_reference = samples[audio_start_sample:audio_end_sample]
        crop_matches_source_slice_exact = bool(np.array_equal(crop, source_slice_reference))
        crop_all_finite = bool(np.all(np.isfinite(crop)))
        crop_sample_count = int(len(crop))
        expected_sample_count = audio_end_sample - audio_start_sample

        audio_path = AUDIO_DIR / f"{unit_id}.wav"
        sf.write(str(audio_path), crop, sr, subtype="FLOAT")

        # Read back and verify a byte-for-byte sample-identical crop (no re-encoding artifacts).
        readback, readback_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        sample_identical_crop = bool(readback_sr == sr and np.array_equal(readback, crop))
        readback_sample_count = int(len(readback))

        audio_sha256 = sha256_of_file(audio_path)

        prior_overlap_note = (
            f"audio_start dips {pre_roll:.3f}s before this unit's core_start, "
            + (
                f"into the previous unit's ({f'CC-SPB1-{ID_START + i - 1:03d}'}) core content"
                if i > 0
                else f"into the previous batch's '{PREVIOUS_UNIT['id']}' ({PREVIOUS_UNIT['global_label']}) core content (ends {PREVIOUS_UNIT['core_end_seconds']}s)"
            )
            + ". This pre-roll seconds figure must be excluded from any future training-duration total to avoid double counting."
        )

        entry = {
            "id": unit_id,
            "global_label": global_label,
            "source_path": str(SOURCE),
            "source_sha256": SOURCE_SHA256_EXPECTED,
            "source_duration_seconds": source_duration,
            "source_sample_rate": sr,
            "source_channels": 1,
            "core_start_seconds": round(core_start, 6),
            "core_end_seconds": round(core_end, 6),
            "core_duration_seconds": round(core_end - core_start, 6),
            "core_start_sample": core_start_sample,
            "core_end_sample": core_end_sample,
            "pre_roll_seconds": round(pre_roll, 6),
            "pre_roll_overlap_note": prior_overlap_note,
            "audio_start_seconds": round(audio_start, 6),
            "audio_end_seconds": round(audio_end, 6),
            "audio_start_sample": audio_start_sample,
            "audio_end_sample": audio_end_sample,
            "audio_duration_seconds": round(audio_end - audio_start, 6),
            "boundary_selection": {
                "method": (
                    "Local minimum of 30ms-frame/10ms-hop RMS energy within an 18-22s window "
                    "after the previous core end. This is an energy heuristic locating a "
                    "relatively quiet moment, NOT a certified word or speaker boundary. "
                    "RMS minimum alone does not prove a word boundary."
                ),
                **boundary_info[i],
            },
            "audio_path": str(audio_path),
            "audio_sha256": audio_sha256,
            "audio_sample_rate": sr,
            "audio_channels": 1,
            "audio_format": "WAV PCM FLOAT32 (no gain/filter/speed/pitch changes)",
            "verification": {
                "source_hash_unchanged_before_after_decode": (
                    source_sha_before == source_sha_after_decode == SOURCE_SHA256_EXPECTED
                ),
                "crop_matches_source_slice_exact": crop_matches_source_slice_exact,
                "sample_identical_crop": sample_identical_crop,
                "crop_all_finite": crop_all_finite,
                "crop_sample_count_matches_expected": crop_sample_count == expected_sample_count,
                "readback_sample_count_matches_crop": readback_sample_count == crop_sample_count,
            },
            "speaker_review": "pending",
            "text_review": "pending",
            "quality_review": "pending",
            "training_admitted": False,
            "notes": (
                "Human-review unit only, not a mandatory training segmentation boundary. "
                "Contains no independent speaker/phoneme certification; frequency/deeper-voice "
                "cues remain user-provided clues only. Presenting this unit does not by itself "
                "satisfy any 20-valid-pair training gate. This batch does not touch, complete, "
                "or shift the prior accepted (incomplete-ending) global20 transcript."
            ),
            "created_utc": utc_now_iso(),
        }
        entries.append(entry)

        batch = {
            "batch_id": BATCH_ID,
            "owner": OWNER,
            "worktree": WORKTREE,
            "branch": BRANCH,
            "generator_script": str(Path(__file__).resolve()),
            "generated_utc": utc_now_iso(),
            "source_path": str(SOURCE),
            "source_sha256": SOURCE_SHA256_EXPECTED,
            "source_duration_seconds": source_duration,
            "source_sample_rate": sr,
            "source_channels": 1,
            "source_immutable": True,
            "previous_unit": PREVIOUS_UNIT,
            "scope_note": (
                "Next two continuous ~20-second review units (CC-SPB1-022, CC-SPB1-023 / "
                "GLOBAL21, GLOBAL22) starting from core 329.050s onward (end of raw "
                "CC-SPB1-021 / GLOBAL20), ending near natural quiet pauses where possible "
                "(18-22s per unit). Contiguous core intervals, no gaps. Human-review units, "
                "not mandatory training segmentation. training_admitted is false for all "
                "entries. Reaching global22 does not by itself satisfy the 20-valid-pair "
                "training gate; current admitted/valid count remains 4."
            ),
            "entries": entries,
        }
        # Write batch.json BEFORE READY.json so READY's referenced manifest
        # path is always readable the moment readiness is published (root
        # fix carried over from batch-03/04/05's scripts).
        (OUT_DIR / "batch.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        write_ready(entries, status="in_progress" if len(entries) < NUM_UNITS else "complete")

        print(f"prepared {unit_id} ({global_label}): core {core_start:.3f}-{core_end:.3f}s "
              f"(audio {audio_start:.3f}-{audio_end:.3f}s) -> {audio_path.name} "
              f"sample_identical_crop={sample_identical_crop}", flush=True)
        print(f"ABSOLUTE PLAYABLE PATH ({unit_id} / {global_label}): {audio_path.resolve()}", flush=True)

    source_sha_final = sha256_of_file(SOURCE)
    assert source_sha_final == SOURCE_SHA256_EXPECTED, "source mutated during batch preparation"
    print(f"source hash unchanged at end: {source_sha_final}", flush=True)
    print(f"wrote {len(entries)} entries to {OUT_DIR / 'batch.json'}", flush=True)


if __name__ == "__main__":
    main()
