import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import soundfile as sf
import numpy as np
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
batch=ROOT/'data/review-batch-01'
clips=[]
for name in ['claude-part2.json','codex-part3.json']:
    proposal=json.loads((batch/'proposals'/name).read_text(encoding='utf-8'))
    for c in proposal['clips']:
        p=ROOT/c['path']
        assert p.resolve().is_relative_to(ROOT.resolve())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256']
        x,rate=sf.read(p,dtype='float32',always_2d=True)
        assert np.isfinite(x).all() and rate==c['sample_rate'] and abs(len(x)/rate-c['duration_seconds'])<1/rate
        assert c['accepted_for_training'] is False
        c['owner']='Claude Code' if name.startswith('claude') else 'Arya/Codex'
        clips.append(c)
ids=[c['id'] for c in clips];assert len(ids)==len(set(ids))
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'owner':'Arya/Codex','spec_commit':'2443c85702998d31e40057270ccad77bd7e5485f','status':'awaiting_human_speech_and_transcript_review','training_ready':False,'admitted_training_seconds':0,'draft_seconds':sum(c['duration_seconds'] for c in clips),'source_split_policy':'All SBS parts one session. Part4 reserved for clearly labeled within-session evaluation; separate Sakshi session not yet reviewed, never assumed suitable Tamil evaluation. Approved12.5s anchor is development-only. No final train/eval admission yet.','clips':clips}
with (ROOT/'data/manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
first=[next(c for c in clips if c['id']==k) for k in ['cc-sbs2-060000','cx-sbs3-060000']]
lines=['# First speech review — two clips, 40 seconds total','','Assumptions: these are candidate windows from existing interviews. Speaker, background quality, complete word boundaries and ASR draft are not yet approved. Nothing here has entered training.','','For each clip, confirm whether it is SPB alone with no overlapping speech/music, and correct the draft to exactly what he says. Mark any cut-off beginning/end or unclear phrase; we can trim or exclude it. Keep colloquial words rather than formalizing them.','']
for i,c in enumerate(first,1):
    lines += [f'## {i}. {c["id"]} — {c["owner"]}', '',f'![Play {c["id"]}]({(ROOT/c["path"]).as_posix()})','',f'Source: SBS part{c["source_part"]}, {c["start_seconds"]:.2f}–{c["end_seconds"]:.2f} seconds.','', '**Uncorrected machine draft:**','',c['transcript_draft'].strip(),'']
lines += ['Two more20-second candidate windows are retained in the manifest for later review; they do not need to be reviewed at once. The previously approved text beginning **அத** remains unchanged.']
(batch/'FIRST-REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'clips':ids,'draft_seconds':manifest['draft_seconds'],'first_review_ids':[c['id'] for c in first],'training_ready':False}))
