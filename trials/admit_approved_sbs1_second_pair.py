"""Admit the approved next unit's nonoverlapping core and explicit text encoding."""
import hashlib,json,os,unicodedata
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
OUT=DATA/'data/paired-sbs1-002-01'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from transformers import VitsTokenizer
from direct_tts.manifest import validate_entry,find_cross_entry_issues
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
history_path=DATA/'data/human-dictation-sbs1-01/CC-SPB1-002-corrections.json'
history=json.loads(history_path.read_text(encoding='utf-8'));assert history['full_clip_approved'] is True
batch=json.loads((DATA/'data/review-sbs1-claude-batch-01/batch.json').read_text(encoding='utf-8'))
entry=next(e for e in batch['entries'] if e['id']=='CC-SPB1-002')
review=Path(entry['audio_path']);assert sha(review)==entry['audio_sha256']==history['audio_sha256']
source=Path(entry['source_path']);assert sha(source)==entry['source_sha256']
verbatim=history['combined_working_text']
mapping={'Recording theatre':'ரெக்கார்டிங் தியேட்டர்','ஃப்ளோர்':'ப்ளோர்','எம்.ஜி.ஆர்.':'எம் ஜி ஆர்'}
pronunciation=verbatim
for old,new in mapping.items():
    assert pronunciation.count(old)==1;pronunciation=pronunciation.replace(old,new)
tokenizer=VitsTokenizer.from_pretrained(ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam',local_files_only=True)
unsupported={c for c in unicodedata.normalize('NFC',pronunciation)
             if unicodedata.category(c)[0] in 'LM' and c not in tokenizer.get_vocab()}
assert not unsupported
ids=tokenizer(pronunciation)['input_ids'];assert ids and len(ids)<768 and min(ids)>=0 and max(ids)<58
y,sr=sf.read(review,dtype='float32');assert sr==48000 and y.ndim==1 and np.isfinite(y).all()
left=entry['core_start_sample']-entry['audio_start_sample']
right=entry['core_end_sample']-entry['audio_start_sample']
assert left==14400 and right==len(y)
core=y[left:right]
# Core start is the same quiet boundary as the previously accepted clip end.
boundary_rms=float(np.sqrt(np.mean(y[left-1440:left+1440].astype('float64')**2)))
assert boundary_rms<.02, 'Core boundary needs review before removing contextual pre-roll'
z=resample_poly(core,1,3).astype('float32');assert np.isfinite(z).all() and np.max(np.abs(z))<1
manifest_path=DATA/'data/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
assert not any(c['id']=='spb1-reviewed-002-core' for c in manifest['clips'])
OUT.mkdir(exist_ok=False)
audio=OUT/'CC-SPB1-002-core-16k.wav';original_text=OUT/'approved-verbatim.txt';encoded_text=OUT/'pronunciation-text.txt'
sf.write(audio,z,16000,subtype='FLOAT');original_text.write_text(verbatim+'\n',encoding='utf-8');encoded_text.write_text(pronunciation+'\n',encoding='utf-8')
approval_snapshot=OUT/'approval-history-snapshot.json';approval_snapshot.write_bytes(history_path.read_bytes())
encoding={'mapping':mapping,'approved_verbatim_preserved':True,
          'authorization':'Previously authorized transparent separate pronunciation encoding for unchanged Tamil tokenizer; no silent original transcript edits.',
          'limitation':'ஃ is unsupported. ப்ளோர் is an explicit imperfect Tamil-vocabulary approximation of ஃப்ளோர், not proof that English f is preserved. Spoken initials are separated explicitly.',
          'token_count':len(ids),'phonetic_quality':'not established by token coverage'}
(OUT/'pronunciation-map.json').write_text(json.dumps(encoding,ensure_ascii=False,indent=2),encoding='utf-8')
record={'id':'spb1-reviewed-002-core','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_path':str(source),'source_sha256':entry['source_sha256'],'source_group':'SBS interview with RaySel, recorded 2014',
        'start_seconds':entry['core_start_seconds'],'end_seconds':entry['core_end_seconds'],
        'path':str(audio),'sha256':sha(audio),'duration_seconds':len(z)/16000,'sample_rate':16000,'channels':1,
        'review_audio':str(review),'review_audio_sha256':entry['audio_sha256'],'review_pre_roll_removed_seconds':left/sr,
        'boundary_rms':boundary_rms,'boundary_claim':'Shared quiet core boundary; ASR/energy and human review support pairing, not certified phoneme alignment',
        'text':pronunciation,'text_file':str(encoded_text),'text_sha256':sha(encoded_text),
        'approved_verbatim_file':str(original_text),'approved_verbatim_sha256':sha(original_text),
        'pronunciation_map_file':str(OUT/'pronunciation-map.json'),'text_token_count':len(ids),
        'human_approval_file':str(approval_snapshot),'human_approval':history['approval_evidence'],
        'target':'spb_speech','text_review':'reviewed','speaker_review':'reviewed','quality_review':'reviewed',
        'accepted_for_training':True,'split':'train','is_diagnostic_anchor':False,
        'use_scope':'Authorized local SPB TTS experiment; transparent pronunciation encoding; synthetic output labels',
        'processing':'Remove0.300s overlapping review pre-roll; core116.760–128.235; scipy resample_poly1/3 to16k. No gain/EQ/pitch/speed edits.',
        'new_training_run_started':False}
assert not validate_entry(record)
train=[c for c in manifest['clips'] if c.get('accepted_for_training') and c.get('split')=='train']+[record]
assert len(train)==3 and not find_cross_entry_issues(train)
(OUT/'pairing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
backup=DATA/'data/manifest-before-third-approved-pair.json';assert not backup.exists();backup.write_bytes(manifest_path.read_bytes())
manifest['clips'].append(record)
manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(),admitted_training_pairs=3,
                admitted_training_seconds=sum(c['duration_seconds'] for c in train),
                status='three_approved_paired_clips; ten_clip_combined_run_milestone_pending',
                next_combined_training_minimum_accepted_pairs=10,new_training_run_started=False)
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'accepted_pairs':3,'unique_training_seconds':manifest['admitted_training_seconds'],
                  'new_core_seconds':record['duration_seconds'],'token_count':len(ids),'boundary_rms':boundary_rms,
                  'pre_roll_excluded':left/sr,'cross_entry_issues':0,'new_training_run_started':False}))
