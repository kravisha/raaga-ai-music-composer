"""Explicit encoding of approved English words for unchanged Tamil vocabulary."""
import json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910';OUT=DATA/'data/review-batch-01'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from transformers import VitsTokenizer
p=OUT/'cc-sbs2-060000-approved-transcript.txt';verbatim=p.read_text(encoding='utf-8').strip();before=hashlib.sha256(p.read_bytes()).hexdigest()
mapping={'playback singer':'ப்ளேபேக் சிங்கர்','singers':'சிங்கர்ஸ்','qualification':'குவாலிபிகேஷன்'}
pronunciation=verbatim
for old,new in mapping.items():
    assert pronunciation.count(old)==1;pronunciation=pronunciation.replace(old,new)
t=VitsTokenizer.from_pretrained(ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam',local_files_only=True)
ids=t(pronunciation)['input_ids'];assert ids and min(ids)>=0 and max(ids)<58
assert all(not ('a'<=c.lower()<='z') for c in pronunciation)
fp=OUT/'cc-sbs2-060000-pronunciation-text.txt';fp.write_text(pronunciation+'\n',encoding='utf-8')
rec={'created_utc':datetime.now(timezone.utc).isoformat(),'clip_id':'cc-sbs2-060000','verbatim_text_file':str(p),'verbatim_sha256':before,'pronunciation_text_file':str(fp),'pronunciation_sha256':hashlib.sha256(fp.read_bytes()).hexdigest(),'mapping':mapping,'authorization':'Krish permits transparent routine encoding fix in a separate pronunciation field; expose exact transliterations for correction, no repeated transcript approval','tokenizer_coverage':'all IDs inside unchanged58-entry model vocabulary; no English spellings lost silently','token_count':len(ids),'phonetic_acceptance':'not established by token coverage; spellings are explicit provisional pronunciation representations, open to user correction','original_text_unchanged':hashlib.sha256(p.read_bytes()).hexdigest()==before,'training_started':False}
(OUT/'cc-sbs2-060000-pronunciation-map.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rec,ensure_ascii=True))
