"""Prepare the newly human-approved source1 clip; retain prior approved data."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
OUT=DATA/'data/paired-sbs1-first-review-01'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from transformers import VitsTokenizer
from direct_tts.manifest import validate_entry
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
correction_file=DATA/'data/human-dictation-sbs1-01/first-review-corrections.json'
correction=json.loads(correction_file.read_text(encoding='utf-8'))
assert correction['full_clip_approved'] is True
review=Path(correction['review_audio']);assert sha(review)==correction['review_audio_sha256']
source=Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-1.mp3')
source_sha='7429c56bc9fcf5e2c69942e4387f8448b0bd5ed99de67638e90de104c96740f7'
assert sha(source)==source_sha
text=correction['combined_working_text']
assert 'நான் ஏதோ ஒரு தெலுங்கு படத்துக்கு டப்பிங் பாடிட்டு இருக்கும்போது' in text
y,sr=sf.read(review,dtype='float32');assert sr==48000 and y.ndim==1 and np.isfinite(y).all()
z=resample_poly(y,1,3).astype('float32');assert np.isfinite(z).all() and np.max(np.abs(z))<1
t=VitsTokenizer.from_pretrained(ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam',local_files_only=True)
ids=t(text)['input_ids'];assert ids and len(ids)<768 and min(ids)>=0 and max(ids)<58
OUT.mkdir(exist_ok=False)
audio=OUT/'SPB1-first-reviewed-16k.wav';textfile=OUT/'approved-transcript.ta.txt'
sf.write(audio,z,16000,subtype='FLOAT');textfile.write_text(text+'\n',encoding='utf-8')
record={'id':'spb1-first-reviewed-105830','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_path':str(source),'source_sha256':source_sha,'source_group':'SBS interview with RaySel, recorded 2014',
        'start_seconds':correction['source_start_seconds'],'end_seconds':correction['source_end_seconds'],
        'path':str(audio),'sha256':sha(audio),'duration_seconds':len(z)/16000,'sample_rate':16000,'channels':1,
        'review_audio':str(review),'review_audio_sha256':sha(review),
        'text':text,'text_file':str(textfile),'text_sha256':sha(textfile),'text_token_count':len(ids),
        'target':'spb_speech','text_review':'reviewed','speaker_review':'reviewed','quality_review':'reviewed',
        'human_approval':correction['approval_evidence'],'human_corrections_file':str(correction_file),
        'human_corrections_sha256_at_admission':sha(correction_file),
        'scope_note':'Best-effort human-approved working transcript; not measured WER or certified phoneme alignment.',
        'boundary_note':'Quiet pre-roll before human-identified first answer, full final word retained per source-context/ASR estimates and practical listening approval. Sentence continues grammatically into next review clip.',
        'use_scope':'Authorized local SPB direct-TTS experiment; preserve synthetic labels',
        'accepted_for_training':True,'split':'train','is_diagnostic_anchor':False,
        'processing':'Full approved review unit resampled48k to16k using scipy resample_poly1/3; no additional trim/gain/EQ/time/pitch change.',
        'new_training_run_started':False}
assert not validate_entry(record)
(OUT/'pairing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
manifest_file=DATA/'data/manifest.json';manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
assert not any(c['id']==record['id'] for c in manifest['clips'])
backup=DATA/'data/manifest-before-second-approved-pair.json';assert not backup.exists();backup.write_bytes(manifest_file.read_bytes())
manifest['clips'].append(record)
train=[c for c in manifest['clips'] if c.get('accepted_for_training') and c.get('split')=='train']
assert len(train)==2 and all(not validate_entry(c) for c in train)
manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(),admitted_training_seconds=sum(c['duration_seconds'] for c in train),
                admitted_training_pairs=len(train),status='two_human_approved_training_pairs; more_review_units_pending',
                new_training_run_started=False)
manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'id':record['id'],'seconds':record['duration_seconds'],'token_count':len(ids),
                  'admitted_pairs':len(train),'admitted_seconds':manifest['admitted_training_seconds'],'new_training_run_started':False}))
