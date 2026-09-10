"""Codex-owned, exact-boundary review crops; no ASR or training."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import soundfile as sf

ROOT=Path('C:/Users/Krish/ClaudeStuff')
OUT=ROOT/'Raaga-SPB-Direct-TTS-20260910/data/review-batch-01'
SOURCE=Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-3.mp3')
EXPECTED='691614afe3830562a00d1653db4847dc8555987aaba4c490989dfbc90cec5136'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

assert sha(SOURCE)==EXPECTED
prior=json.loads((ROOT/'Raaga-Vocal-Quality-20260909/spb-reference-corpus-20260909/speech-window-review.json').read_text(encoding='utf-8-sig'))
records=[]
for start,end,clip_id in [(60.,80.,'cx-sbs3-060000'),(397.66,417.66,'cx-sbs3-397660')]:
    draft=next(x for x in prior['windows'] if x['part']==3 and abs(x['start_seconds']-start)<0.011)
    path=OUT/'clips'/f'{clip_id}.wav'
    if path.exists():raise FileExistsError(path)
    with sf.SoundFile(SOURCE) as audio:
        rate=audio.samplerate;first=round(start*rate);count=round((end-start)*rate)
        audio.seek(first);samples=audio.read(count,dtype='float32',always_2d=True)
    assert len(samples)==count and np.isfinite(samples).all()
    sf.write(path,samples,rate,subtype='FLOAT')
    reloaded,check_rate=sf.read(path,dtype='float32',always_2d=True)
    assert check_rate==rate and np.array_equal(samples,reloaded)
    records.append({'id':clip_id,'source_path':str(SOURCE),'source_sha256':EXPECTED,'source_group':'SBS interview with RaySel, recorded 2014','source_part':3,'start_seconds':start,'end_seconds':end,'duration_seconds':len(samples)/rate,'path':str(path.relative_to(OUT.parent.parent)).replace('\\','/'),'sha256':sha(path),'sample_rate':rate,'channels':samples.shape[1],'peak':float(abs(samples).max()),'finite':True,'crop_verified_sample_identical':True,'processing':'exact source interval decoded to FLOAT WAV; no gain, filter, pitch, time or denoise changes','transcript_draft':draft['machine_transcript'],'transcript_origin':'existing unprompted local Whisper-small 20-second window; not rerun','draft_window_start_seconds':draft['start_seconds'],'draft_window_end_seconds':draft['end_seconds'],'text_review':'pending','speaker_review':'pending','quality_review':'pending','target_speaker':'SPB intended, unverified in this interval','split':'review_only','accepted_for_training':False})
assert sha(SOURCE)==EXPECTED
proposal={'created_utc':datetime.now(timezone.utc).isoformat(),'owner':'Arya/Codex','spec_commit':'2443c85702998d31e40057270ccad77bd7e5485f','source_preserved':True,'training_started':False,'clips':records}
with (OUT/'proposals/codex-part3.json').open('x',encoding='utf-8') as f:json.dump(proposal,f,ensure_ascii=False,indent=2)
print(json.dumps({'clips':[{'id':x['id'],'seconds':x['duration_seconds'],'sha256':x['sha256'],'transcript_draft':x['transcript_draft']} for x in records]},ensure_ascii=True))
