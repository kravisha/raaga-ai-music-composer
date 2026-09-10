"""Fetch only official training metadata; pin before any asset acquisition."""
import json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
OUT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910/assets')
urls={'recipe':'https://api.github.com/repos/ylacombe/finetune-hf-vits/commits/main','checkpoint':'https://huggingface.co/api/models/ylacombe/mms-tts-tam-train?blobs=true'}
result={'created_utc':datetime.now(timezone.utc).isoformat()}
for key,url in urls.items():
    req=urllib.request.Request(url,headers={'User-Agent':'Raaga-local-training-preparation/1.0'})
    with urllib.request.urlopen(req,timeout=30) as r:obj=json.load(r)
    (OUT/f'{key}-remote-metadata.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    result[key]={'url':url,'revision':obj['sha']}
    if key=='checkpoint':result[key]['files']=obj['siblings']
(OUT/'pinned-sources.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
