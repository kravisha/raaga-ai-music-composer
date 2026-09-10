"""Independent source-crop check and provenance-preserving review context index."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,soundfile as sf
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
BATCH=ROOT/'data/review-sbs1-claude-batch-01'
OUT=ROOT/'data/review-sbs1-codex-index-01';OUT.mkdir(exist_ok=False)
manifest=json.loads((BATCH/'batch.json').read_text(encoding='utf-8'))
source=Path(manifest['source_path']);before=hashlib.sha256(source.read_bytes()).hexdigest()
assert before=='7429c56bc9fcf5e2c69942e4387f8448b0bd5ed99de67638e90de104c96740f7'
x,sr=sf.read(source,dtype='float32');assert sr==48000
contexts=[]
for path in [ROOT/'data/review-sbs1-30s-01/review.json',ROOT/'data/context-sbs1-165-195-01/review.json']:
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['source_sha256']==before
    for rec in data['records']:
        for seg in (rec.get('result') or {}).get('chunks',[]):
            ts=seg.get('timestamp');text=seg.get('text','')
            if not ts or ts[0] is None:continue
            start=rec['source_start_seconds']+ts[0]
            end=rec['source_end_seconds'] if ts[1] is None else min(rec['source_end_seconds'],rec['source_start_seconds']+ts[1])
            flags=[]
            if ts[1] is None:flags.append('predicted_end_missing')
            if ts[1] is not None and ts[1]>rec['source_end_seconds']-rec['source_start_seconds']:flags.append('predicted_end_exceeds_context')
            if re.search(r'(.{3,20})\1{4,}',text):flags.append('repetitive_machine_output_unusable')
            contexts.append(dict(source_start_seconds=start,source_end_seconds=end,raw_text=text,
                                 raw_asr_file=str(path),raw_timestamp=ts,flags=flags,status='machine_context_only'))
checked=[];previous=116.76
for entry in manifest['entries']:
    assert abs(entry['core_start_seconds']-previous)<1e-8
    previous=entry['core_end_seconds']
    path=Path(entry['audio_path']);y,yr=sf.read(path,dtype='float32')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['audio_sha256']
    a,b=entry['audio_start_sample'],entry['audio_end_sample']
    assert yr==sr and y.ndim==1 and np.isfinite(y).all() and np.array_equal(y,x[a:b])
    assert abs((b-a)/sr-entry['audio_duration_seconds'])<1e-9
    assert not entry['training_admitted']
    overlapping=[c for c in contexts if c['source_end_seconds']>entry['audio_start_seconds'] and c['source_start_seconds']<entry['audio_end_seconds']]
    checked.append({'id':entry['id'],'audio':str(path),'audio_sha256':entry['audio_sha256'],
                    'core_start_seconds':entry['core_start_seconds'],'core_end_seconds':entry['core_end_seconds'],
                    'audio_start_seconds':entry['audio_start_seconds'],'audio_end_seconds':entry['audio_end_seconds'],
                    'pre_roll_seconds':entry['pre_roll_seconds'],'verified_sample_identical':True,
                    'raw_context_segments':overlapping,'transcript_status':'not exact clip transcript; overlapping machine context only',
                    'training_admitted':False})
assert len(checked)==6 and hashlib.sha256(source.read_bytes()).hexdigest()==before
result={'verified_utc':datetime.now(timezone.utc).isoformat(),'batch_manifest':str(BATCH/'batch.json'),
        'source_sha256':before,'all_six_sample_identical':True,'core_gaps_seconds':0,
        'core_coverage_start_seconds':116.76,'core_coverage_end_seconds':previous,
        'unique_core_seconds':previous-116.76,'pre_roll_rule':'Overlapping review context is never extra training duration',
        'entries':checked}
(OUT/'verified-batch-and-contexts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='entries'}))
