"""Admit approved CC-SPB1-006 and record the twenty-pair collection target.

One-time, local data preparation only. No model training or output overwrites.
"""
import hashlib
import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/Krish/ClaudeStuff')
DATA = ROOT / 'Raaga-SPB-Direct-TTS-20260910'
OUT = DATA / 'data/paired-sbs1-006-01'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from transformers import VitsTokenizer
from direct_tts.manifest import validate_entry, find_cross_entry_issues

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

history_path = DATA/'data/human-dictation-sbs1-01/CC-SPB1-006-corrections.json'
history_bytes = history_path.read_bytes()
history = json.loads(history_bytes.decode('utf-8'))
assert history['dictation_complete'] and history['full_clip_approved']
assert history['working_transcript'].endswith('எழுதுனவரு')
batch = json.loads((DATA/'data/review-sbs1-claude-batch-01/batch.json').read_text(encoding='utf-8'))
entry = next(e for e in batch['entries'] if e['id'] == 'CC-SPB1-006')
review = Path(entry['audio_path'])
assert sha(review) == entry['audio_sha256'] == history['audio_sha256']
assert sha(entry['source_path']) == entry['source_sha256']
text = history['working_transcript']
tokenizer = VitsTokenizer.from_pretrained(ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam', local_files_only=True)
unsupported = {c for c in unicodedata.normalize('NFC', text)
               if unicodedata.category(c)[0] in 'LM' and c not in tokenizer.get_vocab()}
assert not unsupported, 'Unsupported letters need an explicit reviewed pronunciation mapping'
ids = tokenizer(text)['input_ids']
assert ids and len(ids) < 768 and min(ids) >= 0 and max(ids) < 58
y, sr = sf.read(review, dtype='float32')
assert sr == 48000 and y.ndim == 1 and np.isfinite(y).all()
left = entry['core_start_sample'] - entry['audio_start_sample']
assert left == 14400 and len(y) == entry['core_end_sample'] - entry['audio_start_sample']
boundary_rms = float(np.sqrt(np.mean(y[left-720:left+720].astype('float64')**2)))
end_rms = float(np.sqrt(np.mean(y[-1440:].astype('float64')**2)))
assert boundary_rms < .02 and end_rms < .02
z = resample_poly(y[left:], 1, 3).astype('float32')
assert np.isfinite(z).all() and np.max(np.abs(z)) < 1
manifest_path = DATA/'data/manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes.decode('utf-8'))
train = [c for c in manifest['clips'] if c.get('accepted_for_training') and c.get('split') == 'train']
assert len(train) == 3 and not any(c['id'] == 'spb1-reviewed-006-core' for c in manifest['clips'])
OUT.mkdir(exist_ok=False)
audio = OUT/'CC-SPB1-006-core-16k.wav'
text_file = OUT/'approved-verbatim.txt'
approval_file = OUT/'approval-history-snapshot.json'
sf.write(audio, z, 16000, subtype='FLOAT')
text_file.write_text(text+'\n', encoding='utf-8')
approval_file.write_bytes(history_bytes)
reread, rate = sf.read(audio, dtype='float32')
assert rate == 16000 and np.array_equal(reread, z)
record = {
    'id': 'spb1-reviewed-006-core', 'created_utc': datetime.now(timezone.utc).isoformat(),
    'source_path': entry['source_path'], 'source_sha256': entry['source_sha256'],
    'source_group': 'SBS interview with RaySel, recorded 2014',
    'start_seconds': entry['core_start_seconds'], 'end_seconds': entry['core_end_seconds'],
    'path': str(audio), 'sha256': sha(audio), 'duration_seconds': len(z)/16000,
    'sample_rate': 16000, 'channels': 1, 'finite': True,
    'text': text, 'text_file': str(text_file), 'text_sha256': sha(text_file),
    'approved_verbatim_file': str(text_file), 'approved_verbatim_sha256': sha(text_file),
    'pronunciation_mapping': 'None; exact human-approved Tamil working text is supported by the unchanged tokenizer.',
    'text_token_count': len(ids), 'human_approval_file': str(approval_file),
    'human_approval': history['approval_evidence'],
    'target': 'spb_speech', 'text_review': 'reviewed', 'speaker_review': 'reviewed',
    'quality_review': 'reviewed', 'accepted_for_training': True, 'split': 'train',
    'is_diagnostic_anchor': False, 'review_audio': str(review),
    'review_audio_sha256': entry['audio_sha256'], 'review_pre_roll_removed_seconds': left/sr,
    'boundary_rms': boundary_rms, 'end_30ms_rms': end_rms,
    'boundary_claim': 'Human-reviewed working match with quiet source boundaries; not certified phoneme alignment.',
    'processing': 'Remove0.300s contextual pre-roll; resample48k to16k using scipy resample_poly1/3. No gain/EQ/pitch/speed edits.',
    'use_scope': 'Authorized local SPB speech adaptation experiment; generated outputs labeled synthetic.',
    'new_training_run_started': False,
}
assert not validate_entry(record)
train.append(record)
assert not find_cross_entry_issues(train)
write_json(OUT/'pairing.json', record)
with (DATA/'data/manifest-before-fourth-approved-pair.json').open('xb') as f:
    f.write(manifest_bytes)
manifest['clips'].append(record)
manifest.update(admitted_training_pairs=len(train), admitted_training_seconds=sum(c['duration_seconds'] for c in train),
                updated_utc=datetime.now(timezone.utc).isoformat(),
                status='four_approved_paired_clips; twenty_clip_combined_run_milestone_pending',
                next_combined_training_minimum_accepted_pairs=20, new_training_run_started=False,
                current_review_id='CC-SPB1-007',
                current_review_status='Next global eighth007 verified and delivered; no dictation yet. Seventh006 accepted and paired;003tail,004full review and005pairing remain pending.')
write_json(manifest_path, manifest)
history.update(training_admitted=True, paired_derivative_id=record['id'],
               paired_derivative_file='data/paired-sbs1-006-01/pairing.json',
               status='human_approved_working_text_and_paired_core_admitted',
               pairing_review_status='Admitted10.335s core; source/audio/text hashes, token coverage, quiet boundaries and no cross-entry conflicts verified; human working acceptance preserved.')
write_json(history_path, history)
print(json.dumps({'accepted_pairs':len(train), 'accepted_audio_seconds':manifest['admitted_training_seconds'],
                  'new_core_seconds':len(z)/16000, 'token_count':len(ids), 'cross_entry_issues':0,
                  'collection_target':20, 'new_training_run_started':False}))
