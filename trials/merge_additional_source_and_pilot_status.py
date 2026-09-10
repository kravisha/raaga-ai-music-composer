"""Owner-only merge of verified draft delivery and actual pilot status."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import soundfile as sf
import numpy as np
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
p=ROOT/'data/manifest.json';backup=ROOT/'data/manifest-before-batch02-and-pilot-status.json'
proposal_path=ROOT/'data/review-batch-02/proposals/claude-part2.json'
proposal=json.loads(proposal_path.read_text(encoding='utf-8'));item=proposal['clips'][0]
manifest=json.loads(p.read_text(encoding='utf-8'))
assert item['id']=='cc-sbs2-707140' and item['split']=='review_only' and item['accepted_for_training'] is False
assert not any(c['id']==item['id'] for c in manifest['clips'])
audio=ROOT/item['path'];source=Path(item['source_path'])
assert hashlib.sha256(audio.read_bytes()).hexdigest()==item['sha256']
assert hashlib.sha256(source.read_bytes()).hexdigest()==item['source_sha256']
y,sr=sf.read(audio,dtype='float32')
with sf.SoundFile(source) as f:
    f.seek(round(item['start_seconds']*f.samplerate));x=f.read(round(20*f.samplerate),dtype='float32')
assert sr==48000 and y.shape==(960000,) and np.isfinite(y).all()
error=float(np.max(np.abs(x-y)));assert error<1e-6
item.update(independently_verified_utc=datetime.now(timezone.utc).isoformat(),
            independent_crop_max_abs_error=error,independent_sample_identical=bool(np.array_equal(x,y)),
            proposal_path=str(proposal_path),proposal_created_utc_status='unverified author placeholder; do not treat midnight as observed creation time')
assert not backup.exists();backup.write_bytes(p.read_bytes())
manifest['clips'].append(item)
manifest.update(updated_utc=datetime.now(timezone.utc).isoformat(),status='two_explicit_tiny_pilots_complete; broader_corpus_review_pending',
                training_started=True,training_ready=False,full_corpus_training_ready=False,
                draft_seconds=80.,unique_review_candidate_source_seconds=100.,
                readiness_blockers=['Full-corpus evaluation speech/text not yet approved; source1 human corrections partial'],
                validator_status='Integrated83tests passed including65author manifest tests,4independent regressions and14score tests',
                pilots=[{'folder':'pilot-one-utterance-01','steps':30,'status':'runnable; human-reported pronunciation regression; not accepted'},
                        {'folder':'pilot-fixed-text-duration-02','steps':30,'status':'runnable; frozen text/duration verified; final word still ASR-misread; human listening pending'}],
                pilot_scope='Explicitly authorized one-utterance engineering experiments; no held-out speech-quality/generalization claim',
                duration_accounting='First20s parent and18.686s training derivative overlap; count once. Four other20s source candidates pending. All listening/context copies excluded from training duration.')
p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'new_draft':item['id'],'sample_identical':item['independent_sample_identical'],'draft_seconds':manifest['draft_seconds'],'training_pair_seconds':manifest['admitted_training_seconds']}))
