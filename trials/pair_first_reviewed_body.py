"""Pair the approved first text with its complete-clause audio body."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/Krish/ClaudeStuff')
DATA = ROOT / 'Raaga-SPB-Direct-TTS-20260910'
REVIEW = DATA / 'data/review-batch-01'
OUT = DATA / 'data/paired-speech-01'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import VitsTokenizer

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

audio = REVIEW / 'clips/cc-sbs2-060000.wav'
verbatim_path = REVIEW / 'cc-sbs2-060000-approved-transcript.txt'
pronunciation_path = REVIEW / 'cc-sbs2-060000-pronunciation-text.txt'
expected = {audio:'daa75f46bf8a976038b60157bed4dce7b320d5ed3fccade67e862c9668673e28',
            verbatim_path:'662ea13e9c8c1232e78016bab197f36ef91f04d1a8c9297a37e634fd7a08fdb9',
            pronunciation_path:'0705d4ebc7f6ec76abfa115a029207506dde797be0f1d719b79b960e949c8e11'}
assert all(digest(path)==sha for path,sha in expected.items())
verbatim = verbatim_path.read_text(encoding='utf-8').strip()
pronunciation = pronunciation_path.read_text(encoding='utf-8').strip()
suffix = ' எனக்குள்ள ஒரு…'
assert verbatim.endswith(suffix) and pronunciation.endswith(suffix)
body_verbatim, body_pronunciation = verbatim[:-len(suffix)], pronunciation[:-len(suffix)]
x, rate = sf.read(audio, dtype='float32')
assert rate == 48000 and x.ndim == 1 and len(x) == 20*rate and np.isfinite(x).all()
# Word-time estimates place the preceding clause end at source78.54s and
# the incomplete next phrase onset at79.08s. Select a quiet 20ms frame
# wholly inside that gap, then the smallest-amplitude sample near its centre.
width = round(.020*rate)
centres = range(round(18.65*rate), round(18.96*rate)+1, round(.005*rate))
measurements = [(float(np.sqrt(np.mean(x[c-width//2:c+width//2].astype('float64')**2))), c)
                for c in centres]
rms, centre = min(measurements)
radius = round(.001*rate)
end_sample = centre-radius+int(np.argmin(np.abs(x[centre-radius:centre+radius+1])))
body = x[:end_sample]
resampled = resample_poly(body, 1, 3).astype('float32')
assert np.isfinite(resampled).all() and np.max(np.abs(resampled)) < 1
t = VitsTokenizer.from_pretrained(ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam', local_files_only=True)
ids = t(body_pronunciation)['input_ids']
assert ids and min(ids)>=0 and max(ids)<58
OUT.mkdir(exist_ok=False)
body_path = OUT/'cc-sbs2-060000-body-16k.wav'
body_text = OUT/'cc-sbs2-060000-body-pronunciation.txt'
body_verbatim_file = OUT/'cc-sbs2-060000-body-verbatim.txt'
sf.write(body_path, resampled, 16000, subtype='FLOAT')
body_text.write_text(body_pronunciation+'\n', encoding='utf-8')
body_verbatim_file.write_text(body_verbatim+'\n', encoding='utf-8')
record = {
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'id':'cc-sbs2-060000-body', 'parent_clip_id':'cc-sbs2-060000',
    'source_path':'C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-2.mp3',
    'source_sha256':'1833b914cd5a12a997d70d9c6e74cbff4113e9161131b1d5f6de0a2a265c6066',
    'source_group':'SBS interview with RaySel, recorded 2014',
    'parent_clip_path':str(audio), 'parent_clip_sha256':expected[audio],
    'start_seconds':60.0, 'end_seconds':60+end_sample/rate,
    'parent_end_sample_exclusive':end_sample,
    'duration_seconds':len(resampled)/16000, 'path':str(body_path), 'sha256':digest(body_path),
    'sample_rate':16000, 'channels':1, 'peak':float(np.max(np.abs(resampled))), 'finite':True,
    'text':body_pronunciation, 'text_file':str(body_text), 'text_sha256':digest(body_text),
    'body_verbatim_file':str(body_verbatim_file), 'body_verbatim_sha256':digest(body_verbatim_file),
    'approved_full_verbatim_file':str(verbatim_path), 'approved_full_verbatim_sha256':expected[verbatim_path],
    'pronunciation_map_file':str(REVIEW/'cc-sbs2-060000-pronunciation-map.json'),
    'token_count':len(ids), 'required_max_tokens_length':768,
    'target':'spb_speech', 'text_review':'reviewed', 'speaker_review':'reviewed', 'quality_review':'reviewed',
    'review_basis':'Krish listened, corrected the SPB clip, and approved the approximate working transcript: Good. It is the closest I can get you at this point of time. Let us run with this.',
    'confidence':'Human-approved qualitative working match; no measured WER or phoneme-alignment certification.',
    'use_scope':'Local SPB speech adaptation experiment authorized by Krish; synthetic outputs labelled.',
    'accepted_for_training':True, 'split':'train', 'is_diagnostic_anchor':False,
    'processing':'Crop existing reviewed FLOAT WAV at complete clause; explicit scipy resample_poly 1/3 to16k. No gain, filter, pitch, speed or denoise changes.',
    'removed_terminal_text_from_training_only':suffix.strip(),
    'boundary_evidence':{'word_timing_file':str(DATA/'data/semantic-review-02/first-boundary.json'),
        'estimated_preceding_clause_end_source_seconds':78.54,
        'estimated_next_phrase_onset_source_seconds':79.08,
        'quiet_frame_rms':rms, 'quiet_frame_centre_source_seconds':60+centre/rate,
        'method':'20ms lowest RMS frame within source78.65–78.96; smallest-amplitude sample within1ms of frame centre. ASR boundaries are estimates.'},
    'training_started':False,
}
assert all(digest(path)==sha for path,sha in expected.items())
record['original_files_preserved'] = True
(OUT/'pairing.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:record[k] for k in ['id','path','sha256','end_seconds','duration_seconds','token_count','original_files_preserved']}, ensure_ascii=True))
