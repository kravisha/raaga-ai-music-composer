"""Bounded expanded speech-dataset preparation for the collected-data direct-SPB-TTS
trial authorized in docs/spec/SPB_DIRECT_TTS_ADAPTATION.md revision3 (commit
2a6138b37aa141db694b6e488a2bfb52f5e3c239).

Actual Claude Code's exclusive scope: this script, its focused test, and the newly
claimed output directory data/collected-speech-candidates-01/ under the common root.
No old artifact (human corrections, master manifest/cursor/checkpoint, prior pairing
folders, prior model or Codex's trials/run_spb_collected_pilot.py) is modified.

GPU/ASR note: the one authorized bounded local Whisper-small placement/alignment pass
already ran once (GPU float16, batch1, CPU decode) outside this script during
preparation, covering source1 420.305-752.0s to locate the external Tamil-introduction
and Sridhar/MSV passages, plus short per-clip passes on the existing CC-SPB1-013/015/021
review crops to find safe complete-clause cut points. Its raw evidence is embedded below
(ASR_LOCATOR_EVIDENCE / PER_CLIP_ASR_EVIDENCE) and re-emitted into the output directory
for transparency; this script itself performs no further ASR/GPU work (no repeated
sweep), only deterministic audio cropping/resampling and RMS-based boundary confirmation
from those already-obtained timestamps. gpu_released=true throughout.
"""
from __future__ import annotations

import hashlib
import json
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path("C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910")
OUT = ROOT / "data" / "collected-speech-candidates-01"
AUDIO_OUT = OUT / "audio"
TEXT_OUT = OUT / "text"

SOURCE1 = Path("C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-1.mp3")
SOURCE1_SHA256 = "7429c56bc9fcf5e2c69942e4387f8448b0bd5ed99de67638e90de104c96740f7"
SOURCE1_DURATION_S = 829.632
SOURCE2 = Path("C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-2.mp3")
SOURCE2_SHA256 = "1833b914cd5a12a997d70d9c6e74cbff4113e9161131b1d5f6de0a2a265c6066"

SPEC_COMMIT = "2a6138b37aa141db694b6e488a2bfb52f5e3c239"
PRIOR_RELEASE = "ae6f914d9ba4f168dfc9edca6b4ed5aed80a0037"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Embedded evidence from the one bounded local Whisper-small pass (already run,
# GPU float16 batch1; CPU mp3 decode via soundfile). Reproduced verbatim here for
# transparency; this script does not re-invoke the model.
# ---------------------------------------------------------------------------

ASR_LOCATOR_PASS = {
    "purpose": "Locate the external Tamil-introduction and Sridhar/MSV passages beyond "
               "the previously mapped core401.200s boundary, and confirm the "
               "749.8-770s sung-tail boundary already noted by the user.",
    "model": "local openai/whisper-small, GPU float16, batch1, num_beams=1, "
             "language=tamil, task=transcribe, max_new_tokens=384",
    "runtime": "C:/Users/Krish/ClaudeStuff/Raaga-Singing-Experiment-20260908/.venv/Scripts/python.exe",
    "processor_path": "C:/Users/Krish/ClaudeStuff/Raaga-Vocal-Quality-20260909/whisper-processor",
    "model_path": "C:/Users/Krish/ClaudeStuff/Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small",
    "source_path": str(SOURCE1),
    "source_sha256": SOURCE1_SHA256,
    "network_used": False,
    "audio_uploaded": False,
    "windows_20s_non_overlapping_source_seconds": [420.305, 752.0],
    "elapsed_seconds_approx": 106.1,
    "precision_note": "ASR text/timestamps are coarse locator/consistency evidence, not "
                       "certified word alignment.",
    "chunk_results": [
        {"lo": 420.305, "hi": 440.305, "text": "பாட பாட ஐரம் இலவே வா ஓர் ஐரம் இலவே வா இதழோரம் சுவைத் தேட புது பாடல் பாட பாட ஐரம் இலவே வா ஓர் ஐரம் இலவே வா",
         "judgment": "Sung/music passage (song ஆயிரம் நிலவே வா, also referenced textually at the end of CC-SPB1-022); excluded, out of speech scope."},
        {"lo": 440.305, "hi": 460.305, "text": "இது, எஸ்பி ஏஸ் தமிழ். ஆஸ்திரேலியா முழுவதும் ஒளிக்கும் ஒரே தமிழ் ஒலிபரப்பு. மறைந்த பிரபல பின்னணி பாடகர், எஸ்பி பாலசுப்ரமணியம்",
         "judgment": "Other-speaker content: a radio-station (SBS Tamil, Australia) announcer segment, not SPB's own speech; excluded."},
        {"lo": 460.305, "hi": 480.305, "text": "எதுவர்கள் இதுவர்கள் (repeated, low-confidence)",
         "judgment": "Low-confidence/garbled ASR; unclear content, likely still announcer/transition material; excluded, unreviewed."},
        {"lo": 480.305, "hi": 500.305, "text": "முழிகளுக்குமேல் நீங்கள் பாடர்கள் பாட்டுன்ன ஒரு மாபரும் கலைந்தர். இந்த முழிகளுக்கு எல்லாம் பங்க வராமல் அதன் தனி தன்மையோட",
         "judgment": "Unclear, likely continuing other-speaker/announcer content about languages/singers; excluded, unreviewed."},
        {"lo": 500.305, "hi": 520.305, "text": "நா, நா, இதையெல்லாம் அற்புதமாக உச்சரிக்கும் அந்த உச்சரிச்சி பாட்டுரு, மாபரும் பாடக்கிறீர்கள். இந்த திரன் எப்படி கை வரப்ப",
         "judgment": "Unclear, transitional; excluded, unreviewed."},
        {"lo": 520.305, "hi": 540.305, "text": "ஏனும் மிகவும் அவலாக இருந்த ஒரு நாவர் பரணிக்குமார் அப்படின் சொல்லி அவர் பெரிய பப்பிலிச்சிடி ஆர்ட்டிஸ்ட் இந்த போஸ்ட்டர் பப்பிலிச்சிடி எல்லாம் பண்ணுவார்...",
         "judgment": "Strong textual match to the human-supplied external Tamil-introduction "
                     "draft (பரணி குமார் / பப்ளிசிட்டி ஆர்ட்டிஸ்ட் / போஸ்டர் பப்ளிசிட்டி). Refined below with a focused re-pass."},
        {"lo": 540.305, "hi": 560.305, "text": "இங்கு மிக்டிரட்டுத்தான் உன்னைப் பற்றி சொல்லவேண்டும் அடுத்து நாளேங்கள் இங்கு அவ்விஸ்க வாங்கும் என்று சித்திராலாய் அவ்விஸ்க அழ...",
         "judgment": "Strong textual match to the opening of the human-supplied Sridhar/MSV "
                     "main_input draft (அடுத்த நாள் ஆபீஸுக்கு வாங்க சித்ராலயா ஆபீஸுக்கு). Confirms main_input begins here."},
        {"lo": 560.305, "hi": 580.305, "text": "இது ஏன் அர்ஷேர் சிரியார்ச்சில் செய்துகொண்டு இருந்தார்கள் அதெல்லாம் நடக்கும் வரலாங்க்கு இருந்து அதெல்லாம் ஆனதுக்கப் பிறக...",
         "judgment": "Continues matching main_input (ஆர்கெஸ்ட்ராவோடு ... அதெல்லாம் நடக்க வரையிலும் அங்கே இருந்துட்டு)."},
        {"lo": 580.305, "hi": 600.305, "text": "முதலில் ஒரு இந்திப் பாட்டு பாட்டுன்ன, ஏதாவது தமிழி பாடுங்க அப்படினார். UPBS சார்...",
         "judgment": "Continues matching main_input (முதலில் ஒரு ஹிந்தி பாட்டு பாடுனேன் ... உ பி பி எஸ் சார்)."},
        {"lo": 600.305, "hi": 620.305, "text": "விஸ்வநாதன் சார் பாட்டு முடிந்ததுக்காப்பிற்குத் தம்பி ரொம்ப நல்ல பாட்டிருங்க ஆனா தமிழ் நீங்கிந்த மாதிர் உச்சரிச்சின்னா...",
         "judgment": "Continues matching main_input closely."},
        {"lo": 620.305, "hi": 640.305, "text": "அழகாக உச்சரிக்குக் குடிய அழவுக்கு தமிழ் கத்திட்டு வாங்க, உங்களுக்கு நிச்சயம் அவகாசம் இருக்கின்றன்று சொன்னார். ஏற்றாலும்...",
         "judgment": "Matches main_input's truncated ending (அழகாக உச்சரிக்க கூடிய அளவுக்கு...) "
                     "then continues into content NOT present in either supplied fragment: "
                     "this is the missing bridge, ASR-only, not human-reviewed."},
        {"lo": 640.305, "hi": 660.305, "text": "அது தைரியம் வந்ததற்கு அவருக்குட்ட போனா அப்பு முதல் முதலாக ஒரு அவகாசம் கொடுத்தார். அந்த பணம்(படம்) வெளிவே வரில்ல, ஓட்டல் ரம்பா...",
         "judgment": "Bridge continues (ASR-only) then matches the start of the human-supplied "
                     "transcript_tail draft (முதலாக ஒரு அவகாசம் கொடுத்தார் ... ஹோட்டல் ரம்பா)."},
        {"lo": 660.305, "hi": 680.305, "text": "நித்திரையம் விள்ணாட்சின் அந்த பழம் வெளி வருல்லாம் ... முதலவுகாசம் இப்படியாட்சின் பார்த்தேன் அதற்குப் பிறகு",
         "judgment": "Continues matching transcript_tail (நித்திரையும் வீணாச்சு ... முதல் அவகாசம் இப்பிடி ஆச்சேன்னு பாத்தேன்)."},
        {"lo": 680.305, "hi": 700.305, "text": "என் தாய் முடி மலையாளமையா எனக்கு தமிழ்தான் நிறிய வகாசங்கள் வந்தது அதனால் நான் தமிழ் நண்றாக கற்றுக்கொண்டேன் அதற்குப் பிறந...",
         "judgment": "Strong match to transcript_tail (என் தாய் மொழி மலையாளம் ... தமிழ் நன்றாக கற்றுக் கொண்டேன்)."},
        {"lo": 700.305, "hi": 720.305, "text": "அதுதான் நீ பாவத்துடை பாடல முடியும் அப்படி என்று சொன்னார். அதே மாதிரி எந்த முடி பாடினாலும் நான் அந்த முடி ஒரு பண்டமெண்டல...",
         "judgment": "Matches transcript_tail (பாவத்தோட பாட முடியும் ... ஃபண்டமெண்டல்ஸ்)."},
        {"lo": 720.305, "hi": 740.305, "text": "அவனுடைய பாடல் சொல்லி பாடி நான் கேட்டு, நிறைய தெரியவார்கள் ஏற்பார்கள் செய்துகொண்டும் போகிறார்கள். அவர் பரவாயில்லையும் போ...",
         "judgment": "Matches transcript_tail; corroborated independently by the pre-existing "
                     "speech-window-review.json part1 window 709.63-729.63 machine_transcript "
                     "(பெங்காலி/உரியா/படகாய், கவிஞர்/பாடச் சொல்லி), which was already on disk "
                     "before this session and gave the first anchor for this region."},
        {"lo": 740.305, "hi": 752.0, "text": "அந்த முழிக்காரங்கள் நம்மக்குப் பாட்டு கொடுக்கும்போது, அப்படி என்று சொல்லிக் கொடுத்தாங்கள். அந்தபக்தி உடன்தான் எல்லாமுடை...",
         "judgment": "Matches transcript_tail's final clause (மொழிக்காரங்க ... பக்தியுடன் தான் எல்லா "
                     "மொழியும் நான் கற்று பாடுகிறேன்) right up to the point where the human-supplied "
                     "draft itself reports a possible sung/music tail (லா லா லா லா லா), consistent with "
                     "the user's note that the sung section sits at approximately 749.8-770s."},
    ],
}

TAMIL_INTRO_FOCUSED_ASR = {
    "window_source_seconds": [510.0, 545.0],
    "text": "தமிழ் பாட்டிற்காக நான் முயற்சி செய்யும்போது என்ன முதல் முதல் ஆக, தமிழ் துரைக்கு அறிமுகம் "
            "செய்யணம் என்று மிகவும் அவலாக இருந்த ஒரு நாவர் பரணிக்குமார் அப்படின் சுலியில் அவர் பெரிய "
            "பப்பிலிச்சிடி ஆர்ட்டிஸ்ட் என்ற போச்ட்டில் பப்பிலிச்சிடி எல்லாம் சென்றுவார். அவரும் அவர் "
            "ஏக்க்குன்று கூடானார். அவர் கொண்டு போய் சிரிதர் சார்க்க என்ன அறுமுக்குப் படுத்துனார்.",
    "chunks_absolute_source_seconds": [
        [510.0, 525.0, "தமிழ் பாட்டிற்காக நான் முயற்சி செய்யும்போது என்ன முதல் முதல் ஆக, தமிழ் துரைக்கு "
                       "அறிமுகம் செய்யணம் என்று மிகவும் அவலாக இருந்த ஒரு நாவர் பரணிக்குமார் அப்படின் சுலியில்"],
        [525.0, 531.0, "அவர் பெரிய பப்பிலிச்சிடி ஆர்ட்டிஸ்ட் என்ற போச்ட்டில் பப்பிலிச்சிடி எல்லாம் சென்றுவார். "
                       "அவரும் அவர் ஏக்க்குன்று கூடானார்."],
        [531.0, 534.48, "அவர் கொண்டு போய் சிரிதர் சார்க்க என்ன அறுமுக்குப் படுத்துனார்."],
    ],
    "rms_start_boundary": {
        "search_range_source_seconds": [505.0, 512.0],
        "finding": "Continuous, moderate-to-strong energy (announcer content, RMS "
                   "0.05-0.36) through 509.9s, a brief weak blip 510.06-510.36s "
                   "(RMS 0.03-0.16), then a clear >1.1s near-silence 510.36-511.50s "
                   "(RMS <0.005), then strong continuous speech from 511.52s "
                   "(RMS 0.15-0.28) matching the Tamil-introduction sentence.",
        "chosen_cut_source_seconds": 511.10,
        "chosen_cut_rms": 0.00043,
        "judgment": "Cut placed in the unambiguous long silence, after the ambiguous weak "
                    "blip. The >1.1s pause is far longer than a normal mid-sentence breath "
                    "and matches a speaker/segment transition (radio-announcer content ending, "
                    "SPB resuming). Residual uncertainty: the weak blip at 510.06-510.36s is "
                    "not attributed to the target sentence; if it were in fact the faint onset "
                    "of the target word, this candidate's leading edge is very slightly late. "
                    "Recorded as explicit uncertainty rather than guessed.",
    },
    "rms_end_boundary": {
        "search_range_source_seconds": [528.0, 533.0],
        "finding": "Speech energy for ...பப்ளிசிட்டி எல்லாம் பண்ணுவாரு tapers by ~528.70s, "
                   "a clear quiet dip 528.72-529.18s (RMS 0.003-0.007) follows, then energy "
                   "resumes at 529.20s+ for the separate, ASR-only clause அவரும் அவர் "
                   "ஏக்க்குன்று கூடானார் (not part of the human-supplied sentence).",
        "chosen_cut_source_seconds": 528.96,
        "chosen_cut_rms": 0.00248,
        "judgment": "Cut placed in the confirmed post-sentence quiet dip, before the "
                    "unrelated following clause; consistent with the approved text ending "
                    "at 'பண்ணுவாரு.'",
    },
}

PER_CLIP_ASR_EVIDENCE = {
    "CC-SPB1-013": {
        "review_audio": str(ROOT / "data/review-sbs1-claude-batch-03/audio/CC-SPB1-013.wav"),
        "review_audio_sha256": "f985d388c1634c5771760dd7cd73d7b9cf59fed6b4912068e79e982d57ae87d8",
        "asr_chunks_absolute_source_seconds": [
            [222.485, 228.885, "நான் ஒரு சின்ன ரூம்லிருந்து அஞ்சனையிருந்து படித்துக்கொண்டு இருந்து "
                                "எனக்குப் போன்றும் கடையாது யாருக்குப் போன்று சொல்ல வேண்டும் என்று எனக்கும் தெரியாது."],
            [228.885, 234.285, "சே, கரட்டா அந்த கடசி ரிகாடிங்க முந்தின நாள் ரியார்ச்சல் செய்துவிட்டேன்"],
        ],
        "rms_cut_search_source_seconds": [228.4, 229.4],
        "chosen_cut_source_seconds": 228.86,
        "chosen_cut_rms": 0.00146,
        "judgment": "ASR chunk boundary (228.885s) and a confirmed quiet dip (228.82-228.90s, "
                    "RMS<0.006) agree closely; cut placed there, right after ...எனக்கும் தெரியாது. "
                    "and before the unresolved, ambiguous trailing rehearsals clause that "
                    "documented handoff notes flag as overlapping with CC-SPB1-014's opening.",
    },
    "CC-SPB1-015": {
        "review_audio": str(ROOT / "data/review-sbs1-claude-batch-03/audio/CC-SPB1-015.wav"),
        "review_audio_sha256": "54ee3d526557690b196da6d0088f50b338189da3fdafeebc4297f7830911818a",
        "asr_chunks_absolute_source_seconds": [
            [243.295, 247.975, "மஜ ராவர்களுக்குப் பாட்டுப் பாட்டுவதற்காக மிகவும் மிகவும் அவலாக இருப்பாங்க"],
            [247.975, 251.495, "அவரைக் கூட்டும் என்று அவகாசம் கொடுக்கும்போது இந்த மாது இந்த நேரத்தில் நீ"],
        ],
        "rms_cut_search_source_seconds": [246.795, 249.275],
        "chosen_cut_source_seconds": 247.95,
        "chosen_cut_rms": 0.00201,
        "judgment": "Two candidate quiet dips exist (247.42-247.52s and 247.88-248.00s). The "
                    "later one matches the ASR chunk boundary (247.975s) closely and follows a "
                    "further ~0.32s of moderate energy consistent with the multi-syllable word "
                    "இருப்பாங்க tapering to completion; the earlier dip is judged to be a "
                    "mid-word/mid-phrase micro-pause. Choosing the later, ASR-corroborated dip "
                    "avoids the risk of truncating இருப்பாங்க.",
    },
    "CC-SPB1-021": {
        "review_audio": str(ROOT / "data/review-sbs1-claude-batch-05/audio/CC-SPB1-021.wav"),
        "review_audio_sha256": "bd4fbf30eb5f9e2e04b58570351294b5ef9304a71f98f53919f1364201ee9d28",
        "asr_chunks_absolute_source_seconds": [
            [309.005, 312.605, "நான் நிறைய பத்திரிகாடல் எல்லாம் வந்திருந்தார் அவ்வுதான் தெள்விஷ்ஹ்ஹ்டையாது"],
            [312.605, 318.525, "அதே மாதிரி அவர் பிருடியுசர்கள் எல்லார்யும் கூட அவர் அழைத்திருந்தார் அவர்கள் எல்லாம் கூட வந்தார்கள்"],
            [318.525, 323.205, "ரிக்காடிங்க முடிந்தது எல்லார் நல்ல இருக்கு அவ்வு என்று சொல்லி விட்டுவிட்டார்கள்"],
            [323.205, 329.405, "டைரியம் எடுத்துவார்க்கிட்டு நான் கேட்டார் ஐயார் எனக்காக இத்தனை நான் நீங்கள் வீட்டும்"],
        ],
        "focused_repass_absolute_source_seconds": [
            [322.0, 326.2, "தேரியம் எடுத்துவிட்டு நான் கேட்டேன்"],
            [326.2, 329.12, "ஐயார் எனக்காக இத்தனை நாள் நீங்கள் வீட்டும்"],
        ],
        "rms_cut_search_source_seconds": [325.9, 326.5],
        "chosen_cut_source_seconds": 326.15,
        "chosen_cut_rms": 0.00082,
        "judgment": "A focused re-pass isolates the boundary at 326.2s, matching a confirmed "
                    "quiet stretch (325.93-326.45s, RMS<0.002). Cut placed there, right after "
                    "...தைரியம் எடுத்து அவரு கிட்ட கேட்டேன். and before the incomplete, approved-as-"
                    "pending வெயிட்... continuation.",
    },
}


def crop_source(full: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    lo, hi = round(start_s * sr), round(end_s * sr)
    seg = full[lo:hi]
    assert np.isfinite(seg).all()
    return seg


def to_16k(seg48: np.ndarray) -> np.ndarray:
    return resample_poly(seg48, 1, 3).astype(np.float32)


def write_wav16k(path: Path, samples16k: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples16k, 16000, subtype="FLOAT")
    return sha256_file(path)


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    assert sha256_file(SOURCE1) == SOURCE1_SHA256, "source1 hash drift"
    assert sha256_file(SOURCE2) == SOURCE2_SHA256, "source2 hash drift"

    OUT.mkdir(parents=False, exist_ok=False)
    AUDIO_OUT.mkdir(parents=True, exist_ok=False)
    TEXT_OUT.mkdir(parents=True, exist_ok=False)

    # ---------------------------------------------------------------
    # Early inventory/status snapshot, written before any candidate audio
    # is produced, from the human-dictation correction files and manifest.
    # ---------------------------------------------------------------
    early_status = {
        "status": "inventory_complete_processing_candidates",
        "started_utc": now(),
        "spec_commit": SPEC_COMMIT,
        "prior_release": PRIOR_RELEASE,
        "owner": "Actual Claude Code",
        "worktree": "C:/Users/Krish/ClaudeStuff/raaga-wt-claude-spb-source1-batch2",
        "branch": "claude/spb-source1-batch2-20260910",
        "source1_path": str(SOURCE1),
        "source1_sha256": SOURCE1_SHA256,
        "source1_duration_seconds": SOURCE1_DURATION_S,
        "existing_four_valid_admitted_pairs_seconds": 51.4259375,
        "reviewed_source1_clip_inventory": [
            {"id": "spb1-first-reviewed-105830", "core": [105.830, 116.760], "status": "previously_human_approved_pair"},
            {"id": "CC-SPB1-002", "core": [116.760, 128.235], "status": "previously_human_approved_pair"},
            {"id": "CC-SPB1-003", "core": [128.235, 137.390], "status": "excluded: full_clip_approved=false, unclear/unresolved trailing speech"},
            {"id": "CC-SPB1-004", "core": [137.390, 145.985], "status": "excluded: full_clip_approved=false, repeated opening ambiguous"},
            {"id": "CC-SPB1-005", "core": [145.985, 155.290], "status": "excluded: full_clip_approved=false, pending pairing review"},
            {"id": "CC-SPB1-006", "core": [155.290, 165.625], "status": "previously_human_approved_pair"},
            {"id": "CC-SPB1-007", "core": [165.625, 173.650], "status": "excluded: full_clip_approved=false despite complete dictation"},
            {"id": "CC-SPB1-008", "core": [173.650, 183.015], "status": "excluded: WITHDRAWN (008-009 seam evidence); historical only"},
            {"id": "CC-SPB1-009", "core": [183.015, 193.200], "status": "excluded: pairing_blocked, permissions wording uncertain, prior-word seam unresolved"},
            {"id": "CC-SPB1-010", "core": [193.200, 204.475], "status": "excluded: pairing_blocked, opening order/terminal அது unresolved"},
            {"id": "CC-SPB1-011", "core": [204.475, 212.890], "status": "excluded: no clean internal clause boundary; opens/ends mid-continuation; documented overlap with 012"},
            {"id": "CC-SPB1-012", "core": [212.890, 222.785], "status": "excluded: full_clip_approved=false"},
            {"id": "CC-SPB1-013", "core": [222.785, 234.210], "status": "new candidate (first sentence only, 222.785-228.860); remainder excluded (ambiguous trailing rehearsals/014 seam)"},
            {"id": "CC-SPB1-014", "core": [234.210, 243.595], "status": "excluded: full_clip_approved=false, pairing_blocked, opening/tail unresolved"},
            {"id": "CC-SPB1-015", "core": [243.595, 251.890], "status": "new candidate (first sentence only, 243.595-247.950); remainder excluded (incomplete trailing நீ...)"},
            {"id": "CC-SPB1-016-017-20s", "core": [251.890, 272.760], "status": "excluded: full_clip_approved=false, pairing_blocked, opening/middle unresolved"},
            {"id": "CC-SPB1-018-019-20s", "core": [272.760, 290.800], "status": "excluded: full_clip_approved=false, pairing_blocked, several words unverified"},
            {"id": "CC-SPB1-020", "core": [290.800, 309.305], "status": "excluded: full_clip_approved=false, unclarified original bridge"},
            {"id": "CC-SPB1-021", "core": [309.305, 329.050], "status": "new candidate (through கேட்டேன். only, 309.305-326.150); remainder excluded (incomplete trailing வெயிட்...)"},
            {"id": "CC-SPB1-022", "core": [329.050, 348.695], "status": "excluded: full_clip_approved=false, question-variant unresolved, incomplete ஓட... tail"},
            {"id": "CC-SPB1-023", "core": [348.695, 367.260], "status": "new candidate, full clean segment; selected as within-session HOLDOUT"},
            {"id": "SPB1-continuous-after-global22-01 (024+025)", "core": [367.260, 401.200], "status": "excluded: text_review_accepted=false for the combined continuous draft"},
            {"id": "CC-SPB1-026", "core": [401.200, 420.305], "status": "excluded: no text dictated yet"},
            {"id": "(bounded-ASR-located) sung passage", "core": [420.305, 440.305], "status": "excluded: sung/music (ஆயிரம் நிலவே வா), out of speech scope"},
            {"id": "(bounded-ASR-located) other-speaker/unclear", "core": [440.305, 509.900], "status": "excluded: other-speaker radio-announcer content and low-confidence ASR, no supplied SPB text"},
            {"id": "SPB1-external-tamil-introduction-01", "core": [511.100, 528.960], "status": "new candidate, located this session via bounded ASR + RMS boundary evidence"},
            {"id": "(bounded-ASR-located) unclear connective + Sridhar/MSV main_input+bridge+transcript_tail", "core": [528.960, 749.800], "status": "excluded: located via ASR but not safely divisible into ≤20s complete-clause candidates within this bounded pass; explicit missing bridge ~620-650s"},
            {"id": "(user-noted) sung tail", "core": [749.800, 770.000], "status": "excluded: sung/music, out of speech scope per task"},
            {"id": "(unswept remainder)", "core": [770.000, 829.632], "status": "excluded: beyond this session's bounded ASR sweep (420.305-752.0s)"},
        ],
        "concurrent_activity_note": "During this session, a separate directory "
            "data/collected-speech-independent-supplement-01/ appeared under the common "
            "root (owned by another collaborator's concurrent work); it was not read, "
            "used as input, or modified by this script.",
        "note": "This snapshot is written before candidate audio is produced; see "
                "candidates.json and REPORT.md for the final, audio-verified result.",
    }
    write_text(OUT / "status.json", json.dumps(early_status, ensure_ascii=False, indent=2) + "\n")
    write_text(OUT / "asr-locator-evidence.json",
               json.dumps({"locator_pass": ASR_LOCATOR_PASS,
                           "tamil_introduction_focused_repass": TAMIL_INTRO_FOCUSED_ASR,
                           "per_clip_boundary_repasses": PER_CLIP_ASR_EVIDENCE,
                           "gpu_released": True}, ensure_ascii=False, indent=2) + "\n")

    full1, sr1 = sf.read(SOURCE1, dtype="float32")
    assert sr1 == 48000 and full1.ndim == 1

    candidates = []
    exclusions = []

    # ---------------- Old, previously admitted pairs (unchanged, referenced in place) ----------------
    old_pairs = [
        dict(id="cc-sbs2-060000-body",
             path=str(ROOT / "data/paired-speech-01/cc-sbs2-060000-body-16k.wav"),
             sha256="ab60cc1880aac0e6b23415ffd732fc8747601c284d1555a3c92ed6009d016af5",
             text="பாத்திரங்களோட மனப்பாங்கிற்கேத்த மாதிரி பாடக்கூடியவன்தான் ஒரு ப்ளேபேக் சிங்கர். "
                  "நிறைய சிங்கர்ஸ் இருந்தாலும், அப்படிப்பட்ட அவர்களுக்குத்தான் முக்கியத்துவம் "
                  "கிடைக்கிறது. அவர்கள்தான் சக்சஸ் ஆகுறாங்க. அது ஒரு முக்கியமான குவாலிபிகேஷன். "
                  "நாம் மட்டும்… நிறைய பாடகர்கள் ரொம்ப அழகா பாடுகிறாங்க.",
             text_file=str(ROOT / "data/paired-speech-01/cc-sbs2-060000-body-pronunciation.txt"),
             text_sha256="c7d41c80362c208c0bbc3fb007874890ca6bb00e3185fc45d7da5b1b2330962b",
             raw_text_origin_path=str(ROOT / "data/review-batch-01/cc-sbs2-060000-approved-transcript.txt"),
             raw_text_origin_sha256="662ea13e9c8c1232e78016bab197f36ef91f04d1a8c9297a37e634fd7a08fdb9",
             pronunciation_mappings={},
             source_path=str(SOURCE2), source_sha256=SOURCE2_SHA256, source_group="SBS interview with RaySel, recorded 2014 (part2)",
             start=60.0, end=78.68589583333333, duration=18.6859375,
             parent_review_ids=["cc-sbs2-060000"],
             alignment_evidence={"kind": "existing manifest boundary_evidence",
                                  "file": "data/manifest.json#cc-sbs2-060000-body.boundary_evidence",
                                  "method": "20ms lowest-RMS frame within source 78.65-78.96s; human-approved.",
                                  "match_judgment": "Human-approved working match (Krish listened and approved)."},
             quality_evidence={"peak": 0.8270307779312134, "finite": True, "sample_rate": 16000},
             holdout=False),
        dict(id="spb1-first-reviewed-105830",
             path=str(ROOT / "data/paired-sbs1-first-review-01/SPB1-first-reviewed-16k.wav"),
             sha256="98a6be9c9b7dea214aa369487fa174fc94f4b3230cc41492787a10c98d7a19da",
             text="நிறைய சம்பவங்கள் மறக்க முடியாதது. ஆனா எல்லாத்துக்குமே பெரிசா ஒரு விஷயம் தான் நான் "
                  "சொல்லணும். நான் ஏதோ ஒரு தெலுங்கு படத்துக்கு டப்பிங் பாடிட்டு இருக்கும்போது",
             text_file=str(ROOT / "data/paired-sbs1-first-review-01/approved-transcript.ta.txt"),
             text_sha256="a43bf6d2c30ead7576652b6a5fc3ed0055c7e4f4f353a400daf2bb67b1b4a787",
             raw_text_origin_path=str(ROOT / "data/human-dictation-sbs1-01/first-review-corrections.json"),
             raw_text_origin_sha256=None,
             pronunciation_mappings={},
             source_path=str(SOURCE1), source_sha256=SOURCE1_SHA256, source_group="SBS interview with RaySel, recorded 2014 (part1)",
             start=105.83, end=116.76, duration=10.93,
             parent_review_ids=["spb1-first-reviewed-105830"],
             alignment_evidence={"kind": "existing manifest human review",
                                  "file": "data/manifest.json#spb1-first-reviewed-105830",
                                  "method": "Quiet pre-roll trim + human listening approval.",
                                  "match_judgment": "Human-approved (Krish: 'Okay. It's good. Let's go to the second clip.')"},
             quality_evidence={"finite": True, "sample_rate": 16000},
             holdout=False),
        dict(id="spb1-reviewed-002-core",
             path=str(ROOT / "data/paired-sbs1-002-01/CC-SPB1-002-core-16k.wav"),
             sha256="730ba60f1d48f9a9cb25c292482d910d3c7f5a57c5bfca0d68de88141e60b930",
             text="ரெக்கார்டிங் தியேட்டர்ல இருந்து வந்துக்கிட்டு இருக்கிற என்னுடைய ஓசையை, ப்ளோர் "
                  "வெளியே உட்கார்ந்திருந்த எம் ஜி ஆர் அவர்கள் அதைக் கேட்டு, அங்க ஒரு ஆளை "
                  "அனுப்பி, அங்க பையன் யார் என்று கேட்டு வாங்க.",
             text_file=str(ROOT / "data/paired-sbs1-002-01/pronunciation-text.txt"),
             text_sha256="1e1b259d6a5d475e6230d156d3412b07297b4aae4c13eda8f4408ed09aac7f68",
             raw_text_origin_path=str(ROOT / "data/paired-sbs1-002-01/approved-verbatim.txt"),
             raw_text_origin_sha256="46e620c1cf4cf81cef4eb9a263412f9977eeb797b4c10bc4f609259105ffc56c",
             pronunciation_mappings={"ஃப்ளோர்": "ப்ளோர்", "Recording theatre": "ரெக்கார்டிங் தியேட்டர்", "எம்.ஜி.ஆர்.": "எம் ஜி ஆர்"},
             source_path=str(SOURCE1), source_sha256=SOURCE1_SHA256, source_group="SBS interview with RaySel, recorded 2014 (part1)",
             start=116.76, end=128.235, duration=11.475,
             parent_review_ids=["CC-SPB1-002"],
             alignment_evidence={"kind": "existing manifest human review",
                                  "file": "data/manifest.json#spb1-reviewed-002-core",
                                  "method": "0.300s pre-roll removed; shared quiet core boundary.",
                                  "match_judgment": "Human-approved (Krish: 'That's it. Then: now we have three clips.')"},
             quality_evidence={"boundary_rms": 0.0033709633893787825, "finite": True, "sample_rate": 16000},
             holdout=False),
        dict(id="spb1-reviewed-006-core",
             path=str(ROOT / "data/paired-sbs1-006-01/CC-SPB1-006-core-16k.wav"),
             sha256="b1de0e457b95f8aa40430face7918843c2d26f62d39faf9b16ff98892efcc4d3",
             text="மஹாதேவன் சார்தான் இசையமைப்பாளர் அவரும் புகழேந்தி சாரும் வந்திருந்தாரு சுசீலா "
                  "அம்மாவும் வந்திருந்தாரு நானும் போனேன் ஒரு மூன்று நாள் இந்த பாட்டு ரெஹர்ஸல்ஸ் "
                  "பண்ணோம் எழுதுனவரு",
             text_file=str(ROOT / "data/paired-sbs1-006-01/approved-verbatim.txt"),
             text_sha256="5d5cb926aee9d8944a9074a173854d0793b0497c6bcad859c09ce5880a30cc0c",
             raw_text_origin_path=str(ROOT / "data/human-dictation-sbs1-01/CC-SPB1-006-corrections.json"),
             raw_text_origin_sha256=None,
             pronunciation_mappings={},
             source_path=str(SOURCE1), source_sha256=SOURCE1_SHA256, source_group="SBS interview with RaySel, recorded 2014 (part1)",
             start=155.29, end=165.625, duration=10.335,
             parent_review_ids=["CC-SPB1-006"],
             alignment_evidence={"kind": "existing manifest human review",
                                  "file": "data/manifest.json#spb1-reviewed-006-core",
                                  "method": "0.300s pre-roll removed; quiet source boundaries.",
                                  "match_judgment": "Human-approved (Krish approved with corrected joined ending, said Yes)."},
             quality_evidence={"boundary_rms": 0.0007820979284908952, "end_30ms_rms": 0.003276111558766401, "finite": True, "sample_rate": 16000},
             holdout=False),
    ]

    for op in old_pairs:
        actual_sha = sha256_file(Path(op["path"]))
        assert actual_sha == op["sha256"], f"old pair {op['id']} audio hash drift"
        raw_origin_sha = op["raw_text_origin_sha256"]
        if raw_origin_sha is None:
            raw_origin_sha = sha256_file(Path(op["raw_text_origin_path"]))
        candidates.append({
            "id": op["id"],
            "path": op["path"],
            "sha256": op["sha256"],
            "text": op["text"],
            "text_file": op["text_file"],
            "text_sha256": op["text_sha256"],
            "raw_text": op["text"],
            "raw_text_origin_path": op["raw_text_origin_path"],
            "raw_text_origin_sha256": raw_origin_sha,
            "pronunciation_mappings": op["pronunciation_mappings"],
            "source_path": op["source_path"],
            "source_sha256": op["source_sha256"],
            "source_group": op["source_group"],
            "source_start_seconds": op["start"],
            "source_end_seconds": op["end"],
            "duration_seconds": op["duration"],
            "sample_rate": 16000,
            "channels": 1,
            "parent_review_ids": op["parent_review_ids"],
            "text_status": "previously_human_approved",
            "alignment_evidence": op["alignment_evidence"],
            "quality_evidence": op["quality_evidence"],
            "proposed_split": "train",
            "uncertainty": "",
        })

    # ---------------- New candidates within the already-mapped source1 core region ----------------
    new_defs = [
        dict(id="CC-SPB1-013-core-sentence1", start=222.785, end=228.86,
             text="நான் ஒரு சின்ன ரூம்ல இருந்து இன்ஜினியரிங் படிச்சுட்டு இருந்தே எனக்கு போனும் "
                  "கிடையாது யாருக்கும் போன் பண்ணி சொல்லனும்னு எனக்கும் தெரியாது.",
             raw_text="நான் ஒரு சின்ன ரூம்ல இருந்து இன்ஜினியரிங் படிச்சுட்டு இருந்தே எனக்கு ஃபோனும் "
                      "கிடையாது யாருக்கும் ஃபோன் பண்ணி சொல்லனும்னு எனக்கும் தெரியாது.",
             mappings={"ஃபோனும்": "போனும்", "ஃபோன்": "போன்"},
             origin="data/human-dictation-sbs1-01/CC-SPB1-013-corrections.json",
             parent=["CC-SPB1-013"], evidence_key="CC-SPB1-013", holdout=False,
             uncertainty="Boundary derived from local Whisper-small ASR chunk timestamp plus "
                         "20ms-RMS quiet-point search (not human-verified alignment). Text is the "
                         "verbatim leading sentence of Krish's approved CC-SPB1-013 working text "
                         "(full_clip_approved=true at dictation time); the remaining trailing "
                         "clause (சரி கரெக்டா ... rehearsals…) is excluded because handoff notes "
                         "flag an unresolved word-boundary overlap with CC-SPB1-014's opening."),
        dict(id="CC-SPB1-015-core-sentence1", start=243.595, end=247.95,
             text="எம் ஜி ஆர் அவர்களுக்கு பாட்டு பாடறதுக்காக ரொம்ப ரொம்ப ஆவலாக இருப்பாங்க.",
             raw_text="எம் ஜி ஆர் அவர்களுக்கு பாட்டு பாடறதுக்காக ரொம்ப ரொம்ப ஆவலாக இருப்பாங்க.",
             mappings={},
             origin="data/human-dictation-sbs1-01/CC-SPB1-015-corrections.json",
             parent=["CC-SPB1-015"], evidence_key="CC-SPB1-015", holdout=False,
             uncertainty="Boundary derived from local Whisper-small ASR chunk timestamp plus "
                         "20ms-RMS quiet-point search (not human-verified alignment). Text is the "
                         "verbatim leading sentence of Krish's approved CC-SPB1-015 working text; "
                         "the remaining incomplete continuation (...இந்த நேரத்துல நீ…) is excluded, "
                         "its completion depends on the unreviewed CC-SPB1-016-017 draft."),
        dict(id="CC-SPB1-021-core-through-kettén", start=309.305, end=326.15,
             text="நிறைய பத்திரிகையாளர்கள் எல்லாம் வந்திருந்தாங்க. அப்ப தான் டெலிவிஷன் கிடையாது. "
                  "அதே மாதிரி அவர் புரொட்யூசர்ஸ் எல்லாம் எல்லாரையும் அழைத்திருந்தார். அவுங்க "
                  "எல்லாம் கூட வந்தாங்க. ரெக்கார்டிங் முடிஞ்சது. எல்லாரும் நல்லா இருக்கு என்று "
                  "அப்படி சொல்லி ப்ளெஸ் பண்ணாங்க. தைரியம் எடுத்து அவரு கிட்ட கேட்டேன்.",
             raw_text="நிறைய பத்திரிகையாளர்கள் எல்லாம் வந்திருந்தாங்க. அப்ப தான் டெலிவிஷன் கிடையாது. "
                      "அதே மாதிரி அவர் புரொட்யூசர்ஸ் எல்லாம் எல்லாரையும் அழைத்திருந்தார். அவுங்க "
                      "எல்லாம் கூட வந்தாங்க. ரெக்கார்டிங் முடிஞ்சது. எல்லாரும் நல்லா இருக்கு என்று "
                      "அப்படி சொல்லி bless பண்ணாங்க. தைரியம் எடுத்து அவரு கிட்ட கேட்டேன்.",
             mappings={"bless": "ப்ளெஸ்"},
             origin="data/human-dictation-sbs1-01/CC-SPB1-021-corrections.json",
             parent=["CC-SPB1-021"], evidence_key="CC-SPB1-021", holdout=False,
             uncertainty="Boundary derived from a focused local Whisper-small re-pass plus "
                         "20ms-RMS quiet-point search (not human-verified alignment). Text is the "
                         "verbatim leading portion of Krish's approved CC-SPB1-021 working text up "
                         "to and including ...கேட்டேன்.; the incomplete trailing continuation "
                         "(சார் எனக்காக நீங்க இவ்ளோ வெயிட்…) is excluded as explicitly pending "
                         "alignment per the correction record."),
        dict(id="CC-SPB1-023-core", start=348.695, end=367.26,
             text="எம் ஜி ஆருக்குன்னு அவங்கள்லாம் ரொம்ப சந்தோஷப்பட்டு இருப்பாங்க. உனக்கு உடம்பு "
                  "சரி இல்லன்னு அப்படின்னு சொல்லி இந்த பாட்ட வேறு யாருக்காவது கொடுத்து உனக்கு வேற "
                  "ஒரு பாடலும் குடுத்திருக்கலாம். ஆனா படம் பாக்கும் போது உங்க ப்ரெண்ட்ஸ் எல்லாம் "
                  "என்ன நெனப்பாங்கன்னா அவன் பாடி இருப்பான் அது அவருக்கு பிடிக்கல போல இருக்கு "
                  "அதனால்தான் வேற பாட்டுக்கு குடுத்திருக்காரு அப்படின்னு.",
             raw_text="எம் ஜி ஆருக்குன்னு அவங்கள்லாம் ரொம்ப சந்தோஷப்பட்டு இருப்பாங்க. உனக்கு உடம்பு "
                      "சரி இல்லன்னு அப்படின்னு சொல்லி இந்த பாட்ட வேறு யாருக்காவது கொடுத்து உனக்கு "
                      "வேற ஒரு பாடலும் குடுத்திருக்கலாம். ஆனா படம் பாக்கும் போது உங்க ஃப்ரெண்ட்ஸ் "
                      "எல்லாம் என்ன நெனப்பாங்கன்னா அவன் பாடி இருப்பான் அது அவருக்கு பிடிக்கல போல "
                      "இருக்கு அதனால்தான் வேற பாட்டுக்கு குடுத்திருக்காரு அப்படின்னு.",
             mappings={"ஃப்ரெண்ட்ஸ்": "ப்ரெண்ட்ஸ்"},
             origin="data/human-dictation-sbs1-01/CC-SPB1-023-corrections.json",
             parent=["CC-SPB1-023"], evidence_key=None, holdout=True,
             uncertainty="Full core segment used verbatim from the existing established quiet "
                         "core boundaries (348.695/367.260, shared with adjacent review clips); "
                         "no new ASR/RMS trimming was required since the approved text ends "
                         "cleanly with a period and matches the full core span. Selected as the "
                         "within-session holdout: both neighbouring intervals (CC-SPB1-022 and "
                         "the 367.260-401.200 continuous draft) are excluded from training, giving "
                         "large (>20s) gaps to the nearest training interval on both sides."),
        dict(id="SPB1-external-tamil-introduction-01-core", start=511.10, end=528.96,
             text="தமிழ் அப்படினு பாட்றதுக்காக நானு முயற்சு செய்யும் போது என்ன முதல் முதல்லா க "
                  "தமிழ் துறைக்கு அறிமுகம் செய்யணும்னு ரொம்ப ஆவலாக இருந்த ஒருனவர் பரணி குமார் "
                  "அப்படினு சொல்லி அவரு பெரிய பப்ளிசிட்டி ஆர்ட்டிஸ்ட் இந்த போஸ்டர் பப்ளிசிட்டி "
                  "எல்லாம் பண்ணுவாரு .",
             raw_text="தமிழ் அப்படினு பாட்றதுக்காக நானு முயற்சு செய்யும் போது என்ன முதல் முதல்லா க "
                      "தமிழ் துறைக்கு அறிமுகம் செய்யணும்னு ரொம்ப ஆவலாக இருந்த ஒருனவர் பரணி குமார் "
                      "அப்படினு சொல்லி அவரு பெரிய பப்ளிசிட்டி ஆர்ட்டிஸ்ட் இந்த போஸ்டர் பப்ளிசிட்டி "
                      "எல்லாம் பண்ணுவாரு .",
             mappings={},
             origin="data/human-dictation-sbs1-01/SPB1-external-tamil-introduction-01-corrections.json",
             parent=["SPB1-external-tamil-introduction-01"], evidence_key=None, holdout=False,
             uncertainty="Source placement was entirely UNMAPPED before this session "
                         "(source_start/end_seconds were null and status was "
                         "unreviewed_external_draft_mapping_pending). Located this session via a "
                         "bounded local Whisper-small ASR sweep of source1 420.305-752.0s plus a "
                         "focused re-pass and RMS boundary search (see asr-locator-evidence.json); "
                         "text itself has not been compared word-by-word against this audio by "
                         "Krish. A residual ~0.3s ambiguous energy blip immediately precedes the "
                         "chosen start cut and was excluded rather than guessed into the clip; see "
                         "alignment_evidence for the specific judgment."),
    ]

    for nd in new_defs:
        seg48 = crop_source(full1, sr1, nd["start"], nd["end"])
        seg16 = to_16k(seg48)
        out_path = AUDIO_OUT / f"{nd['id']}.wav"
        wav_sha = write_wav16k(out_path, seg16)
        text_path = TEXT_OUT / f"{nd['id']}.ta.txt"
        text_sha = write_text(text_path, nd["text"])
        raw_origin_path = ROOT / nd["origin"]
        raw_origin_sha = sha256_file(raw_origin_path)
        peak = float(np.max(np.abs(seg16))) if seg16.size else 0.0

        if nd["evidence_key"] is not None:
            ev = PER_CLIP_ASR_EVIDENCE[nd["evidence_key"]]
            alignment_evidence = {
                "kind": "local Whisper-small ASR chunk timestamp + 20ms-RMS quiet-point search",
                "review_audio": ev["review_audio"],
                "review_audio_sha256": ev["review_audio_sha256"],
                "asr_chunks_absolute_source_seconds": ev.get("asr_chunks_absolute_source_seconds")
                    or ev.get("focused_repass_absolute_source_seconds"),
                "rms_cut_search_source_seconds": ev["rms_cut_search_source_seconds"],
                "chosen_cut_source_seconds": ev["chosen_cut_source_seconds"],
                "chosen_cut_rms": ev["chosen_cut_rms"],
                "match_judgment": ev["judgment"],
            }
        elif nd["id"].startswith("SPB1-external-tamil-introduction"):
            alignment_evidence = {
                "kind": "local Whisper-small bounded locator sweep + focused re-pass + 20ms-RMS boundary search",
                "locator_window_source_seconds": [420.305, 752.0],
                "focused_repass_window_source_seconds": [510.0, 545.0],
                "start_boundary": TAMIL_INTRO_FOCUSED_ASR["rms_start_boundary"],
                "end_boundary": TAMIL_INTRO_FOCUSED_ASR["rms_end_boundary"],
                "match_judgment": "Strong, near-complete textual correspondence between the "
                                   "ASR transcript of source1 511.10-528.96s and the human-supplied "
                                   "external Tamil-introduction draft (பரணி குமார், பப்ளிசிட்டி "
                                   "ஆர்ட்டிஸ்ட், போஸ்டர் பப்ளிசிட்டி all present in matching order).",
            }
        else:
            alignment_evidence = {
                "kind": "existing established core boundary (shared with adjacent review clips)",
                "source": "data/human-dictation-sbs1-01/CC-SPB1-023-corrections.json",
                "match_judgment": "Full approved working text matches the full established core "
                                   "span with no incomplete trailing clause.",
            }

        candidates.append({
            "id": nd["id"],
            "path": str(out_path),
            "sha256": wav_sha,
            "text": nd["text"],
            "text_file": str(text_path),
            "text_sha256": text_sha,
            "raw_text": nd["raw_text"],
            "raw_text_origin_path": str(raw_origin_path),
            "raw_text_origin_sha256": raw_origin_sha,
            "pronunciation_mappings": nd["mappings"],
            "source_path": str(SOURCE1),
            "source_sha256": SOURCE1_SHA256,
            "source_group": "SBS interview with RaySel, recorded 2014 (part1)",
            "source_start_seconds": nd["start"],
            "source_end_seconds": nd["end"],
            "duration_seconds": round(nd["end"] - nd["start"], 6),
            "sample_rate": 16000,
            "channels": 1,
            "parent_review_ids": nd["parent"],
            "text_status": "source_checked_user_supplied",
            "alignment_evidence": alignment_evidence,
            "quality_evidence": {"peak": peak, "finite": bool(np.isfinite(seg16).all()), "sample_rate": 16000},
            "proposed_split": "holdout" if nd["holdout"] else "train",
            "uncertainty": nd["uncertainty"],
        })

    # ---------------- Exclusion ledger (source1 only; source2 is separately/previously tracked) ----------------
    exclusions = [
        {"interval_source_seconds": [0.000, 105.830], "duration_seconds": 105.830,
         "reason": "Never reviewed/dictated; no supplied transcript exists for this span of source1."},
        {"interval_source_seconds": [128.235, 155.290], "duration_seconds": 27.055,
         "reason": "Covers CC-SPB1-003 (unclear trailing speech), 004 (repeated opening ambiguous), "
                   "005 (pending pairing review); all full_clip_approved=false."},
        {"interval_source_seconds": [165.625, 222.785], "duration_seconds": 57.160,
         "reason": "Covers 007 (full_clip_approved=false), 008 (WITHDRAWN), 009 (pairing_blocked, "
                   "permissions wording uncertain), 010 (pairing_blocked, opening/terminal அது "
                   "unresolved), 011 (no clean internal clause boundary, documented overlap with 012), "
                   "012 (full_clip_approved=false)."},
        {"interval_source_seconds": [228.860, 243.595], "duration_seconds": 14.735,
         "reason": "Remainder of 013 after its trimmed first sentence (ambiguous trailing rehearsals "
                   "clause) plus all of 014 (full_clip_approved=false, pairing_blocked)."},
        {"interval_source_seconds": [247.950, 309.305], "duration_seconds": 61.355,
         "reason": "Remainder of 015 after its trimmed first sentence, plus 016-017 and 018-019 "
                   "combined drafts (both pairing_blocked, opening/middle unresolved) and 020 "
                   "(confirmed fragments but unclarified connecting bridge, full_clip_approved=false)."},
        {"interval_source_seconds": [326.150, 348.695], "duration_seconds": 22.545,
         "reason": "Remainder of 021 after its trimmed portion (incomplete வெயிட்… continuation) plus "
                   "022 (full_clip_approved=false, question-variant unresolved, incomplete ஓட… tail)."},
        {"interval_source_seconds": [367.260, 401.200], "duration_seconds": 33.940,
         "reason": "Continuous global23/24 combined human draft; text_review_accepted=false for the "
                   "whole passage, no per-clip admitted text."},
        {"interval_source_seconds": [401.200, 420.305], "duration_seconds": 19.105,
         "reason": "CC-SPB1-026: audio delivered but no text dictated yet."},
        {"interval_source_seconds": [420.305, 440.305], "duration_seconds": 20.000,
         "reason": "Located via this session's bounded ASR sweep: sung/music passage "
                   "(ஆயிரம் நிலவே வா); out of speech scope."},
        {"interval_source_seconds": [440.305, 509.900], "duration_seconds": 69.595,
         "reason": "Located via this session's bounded ASR sweep: other-speaker radio-announcer "
                   "content (SBS Tamil station identification) and low-confidence/unclear ASR; no "
                   "SPB dictated text exists for this span."},
        {"interval_source_seconds": [509.900, 511.100], "duration_seconds": 1.200,
         "reason": "Ambiguous ~0.3s ASR-only energy blip plus its surrounding transition silence, "
                   "immediately preceding the located Tamil-introduction candidate; excluded rather "
                   "than guessed (see alignment_evidence uncertainty note)."},
        {"interval_source_seconds": [528.960, 531.000], "duration_seconds": 2.040,
         "reason": "Short ASR-only connective clause (அவரும் அவர் ஏக்க்குன்று கூடானார்) between the "
                   "Tamil-introduction candidate and the Sridhar/MSV main_input opening; no human "
                   "text, not admitted."},
        {"interval_source_seconds": [531.000, 749.800], "duration_seconds": 218.800,
         "reason": "Located via this session's bounded ASR sweep as the Sridhar/MSV main_input, an "
                   "explicit missing bridge (~620-650s, content unknown, not reconstructed), and the "
                   "transcript_tail; strong textual correspondence with the human-supplied draft "
                   "confirms placement, but the passage is long, dense, and not yet safely divisible "
                   "into clean <=20s complete-clause candidates within this bounded pass. Next "
                   "technical blocker: needs a dedicated fine-grained (word/phrase-level) alignment "
                   "pass, ideally with human review of the ASR-suggested clause boundaries, beyond "
                   "this session's one-pass budget."},
        {"interval_source_seconds": [749.800, 770.000], "duration_seconds": 20.200,
         "reason": "Sung/music tail, explicitly out of speech scope per this trial's authorization."},
        {"interval_source_seconds": [770.000, 829.632], "duration_seconds": 59.632,
         "reason": "Beyond this session's bounded ASR sweep (420.305-752.0s); unreviewed and "
                   "unlocated remainder of source1."},
    ]

    train = [c for c in candidates if c["proposed_split"] == "train"]
    holdout = [c for c in candidates if c["proposed_split"] == "holdout"]
    train_seconds = round(sum(c["duration_seconds"] for c in train), 6)
    holdout_seconds = round(sum(c["duration_seconds"] for c in holdout), 6)
    source1_candidate_seconds = round(sum(c["duration_seconds"] for c in candidates if c["source_path"] == str(SOURCE1)), 6)
    exclusion_seconds = round(sum(e["duration_seconds"] for e in exclusions), 6)

    coverage = {
        "source1_total_seconds": SOURCE1_DURATION_S,
        "source1_candidate_seconds": source1_candidate_seconds,
        "source1_exclusion_seconds": exclusion_seconds,
        "source1_accounting_closes": abs((source1_candidate_seconds + exclusion_seconds) - SOURCE1_DURATION_S) < 1e-6,
        "source2_candidate_seconds": round(sum(c["duration_seconds"] for c in candidates if c["source_path"] == str(SOURCE2)), 6),
        "aligned_candidate_total_seconds": round(sum(c["duration_seconds"] for c in candidates), 6),
        "train_seconds": train_seconds,
        "holdout_seconds": holdout_seconds,
        "previously_human_approved_seconds": round(sum(c["duration_seconds"] for c in candidates if c["text_status"] == "previously_human_approved"), 6),
        "source_checked_user_supplied_seconds": round(sum(c["duration_seconds"] for c in candidates if c["text_status"] == "source_checked_user_supplied"), 6),
        "note": "Krish's approximately-ten-minute estimate is a target, not a measured "
                "dataset duration; this bounded pass expands admitted material from the prior "
                "51.4259375s (4 pairs) to the totals above, with the remainder of source1 "
                "explicitly itemised in exclusions rather than assumed usable.",
    }

    result = {
        "status": "bounded_pass_complete",
        "spec_commit": SPEC_COMMIT,
        "prior_release": PRIOR_RELEASE,
        "owner": "Actual Claude Code",
        "generated_utc": now(),
        "source_records": {
            "source1": {"path": str(SOURCE1), "sha256": SOURCE1_SHA256, "duration_seconds": SOURCE1_DURATION_S,
                        "sample_rate": 48000, "channels": 1},
            "source2": {"path": str(SOURCE2), "sha256": SOURCE2_SHA256,
                        "note": "Only the one already-admitted old pair (cc-sbs2-060000-body) draws from source2; "
                                "no new source2 material was reviewed this session."},
        },
        "coverage": coverage,
        "candidates": candidates,
        "exclusions": exclusions,
        "gpu_released": True,
    }

    (OUT / "candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_report(result, coverage, train, holdout)
    print(json.dumps({"candidates": len(candidates), "train_seconds": train_seconds,
                       "holdout_seconds": holdout_seconds, "out": str(OUT)}, ensure_ascii=False))


def write_report(result, coverage, train, holdout) -> None:
    lines = []
    a = lines.append
    a("# SPB collected-speech candidate dataset — bounded expansion report")
    a("")
    a(f"Generated {result['generated_utc']}. Spec commit `{SPEC_COMMIT}` (cherry-picked onto prior "
      f"release `{PRIOR_RELEASE}`). Owner: Actual Claude Code, worktree "
      f"`raaga-wt-claude-spb-source1-batch2`, branch `claude/spb-source1-batch2-20260910`.")
    a("")
    a("## Scope and exclusivity")
    a("")
    a("This report and `candidates.json` are produced by the newly added, exclusively-owned "
      "`trials/prepare_spb_collected_dataset.py`, writing only to the newly claimed "
      "`data/collected-speech-candidates-01/`. No old artifact (human-dictation corrections, the "
      "master `manifest.json`, `current-human-review-cursor.json`, `RESUME-CHECKPOINT.json`, prior "
      "pairing folders, prior model checkpoints, or Codex's own trials) was modified. A separate, "
      "concurrently-created `data/collected-speech-independent-supplement-01/` directory appeared "
      "under the common root during this session (another collaborator's own work); it was not read "
      "or used as input here.")
    a("")
    a("## What changed versus the prior four-pair ledger")
    a("")
    a(f"- Previously admitted (unchanged, referenced in place): **4 pairs**, total "
      f"**{coverage['previously_human_approved_seconds']:.4f}s**.")
    a(f"- New candidates prepared this session: **{len(result['candidates']) - 4}**, total "
      f"**{coverage['source_checked_user_supplied_seconds']:.4f}s**.")
    a(f"- Combined admitted candidate duration: **{coverage['aligned_candidate_total_seconds']:.4f}s** "
      f"(train {coverage['train_seconds']:.4f}s + holdout {coverage['holdout_seconds']:.4f}s), up from "
      f"the prior 51.4259375s / 4 pairs.")
    a(f"- Krish's approximately-ten-minute estimate for source1 remains **unverified as a dataset "
      f"duration claim**; it is not treated as measured truth. The bulk of source1 "
      f"({coverage['source1_exclusion_seconds']:.3f}s of {coverage['source1_total_seconds']:.3f}s) "
      "remains excluded, each interval itemised below with a concrete reason.")
    a("")
    a("## Candidates")
    a("")
    a("| id | split | text_status | source | start | end | dur(s) | uncertainty |")
    a("|---|---|---|---|---:|---:|---:|---|")
    for c in result["candidates"]:
        src = "part2" if c["source_path"] == str(SOURCE2) else "part1"
        unc = (c["uncertainty"][:70] + "…") if len(c["uncertainty"]) > 70 else (c["uncertainty"] or "(none)")
        a(f"| {c['id']} | {c['proposed_split']} | {c['text_status']} | {src} | "
          f"{c['source_start_seconds']:.3f} | {c['source_end_seconds']:.3f} | "
          f"{c['duration_seconds']:.3f} | {unc} |")
    a("")
    a("## Holdout")
    a("")
    for c in holdout:
        a(f"- `{c['id']}` (source1 {c['source_start_seconds']:.3f}-{c['source_end_seconds']:.3f}s, "
          f"{c['duration_seconds']:.3f}s). Nearest training interval before: "
          "CC-SPB1-021-core-through-kettén ends 326.150s (gap 22.545s); nearest after: "
          "SPB1-external-tamil-introduction-01-core starts 511.100s (gap 143.840s). Both immediate "
          "neighbours (CC-SPB1-022 and the 367.260-401.200 continuous draft) are excluded from "
          "training, so no adjacent leaked context.")
    a("")
    a("## Coverage accounting (source1)")
    a("")
    a(f"- Source1 total duration: {coverage['source1_total_seconds']:.3f}s")
    a(f"- Admitted candidate duration (train+holdout, source1 only): {coverage['source1_candidate_seconds']:.3f}s")
    a(f"- Excluded duration (itemised below): {coverage['source1_exclusion_seconds']:.3f}s")
    a(f"- Accounting closes exactly: {coverage['source1_accounting_closes']}")
    a("")
    a("## Exclusions (source1, gapless partition with the candidates above)")
    a("")
    a("| interval (s) | dur(s) | reason |")
    a("|---|---:|---|")
    for e in result["exclusions"]:
        lo, hi = e["interval_source_seconds"]
        a(f"| {lo:.3f}-{hi:.3f} | {e['duration_seconds']:.3f} | {e['reason']} |")
    a("")
    a("## Bounded local ASR/alignment pass")
    a("")
    a("One bounded local Whisper-small pass ran during preparation (GPU float16, batch1, CPU mp3 "
      "decode; no network, no audio upload): a 420.305-752.0s locator sweep over source1 in ~20s "
      "windows (~106s wall time), plus short focused re-passes to refine the Tamil-introduction "
      "boundary and the CC-SPB1-013/015/021 sentence cuts. Full transcripts, chunk timestamps and "
      "the RMS-based boundary searches are recorded in `asr-locator-evidence.json`. ASR served only "
      "as locator/consistency evidence; no ASR output replaced or filled in any human-supplied text, "
      "and no invented bridge text was inserted for the explicit missing-bridge gap. GPU ownership "
      "was released after this one pass; `gpu_released: true` throughout the rest of this preparation.")
    a("")
    a("## Next technical blocker")
    a("")
    a("The Sridhar/MSV narrative (source1 approx. 531.0-749.8s, ~218.8s) is now confidently located "
      "and textually corroborated against the human-supplied draft (including independent "
      "corroboration from a pre-existing, previously unrelated `speech-window-review.json` machine "
      "transcript window at 709.63-729.63s), but it is not yet safely divisible into individual "
      "<=20s complete-clause candidates within this session's one-pass ASR budget. A future session "
      "would need: (1) a finer, sentence-by-sentence ASR/alignment pass over this specific span, (2) "
      "human review of the ASR-suggested clause boundaries (the missing bridge around 620-650s in "
      "particular has no supplied text at all and cannot be reconstructed from ASR), and (3) per-"
      "clause English/loanword pronunciation mapping (e.g. 'Director', 'Music', 'Chitralaya') before "
      "any of it could be admitted as training candidates.")
    a("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
