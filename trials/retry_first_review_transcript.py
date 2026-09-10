"""One audio-backed beam-search retry; never silently correct the manifest."""
import os,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
OUT=DATA/'data/review-batch-01/transcript-retry-01';OUT.mkdir(exist_ok=False)
for key,value in {'HF_HOME':str(DATA/'cache/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','MPLCONFIGDIR':str(DATA/'cache/matplotlib')}.items():os.environ[key]=value
import numpy as np,soundfile as sf,torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
torch.set_num_threads(4);torch.set_num_interop_threads(1)
p=DATA/'data/review-batch-01/clips/cc-sbs2-060000.wav'
before=hashlib.sha256(p.read_bytes()).hexdigest()
status={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running','input':str(p),'input_sha256':before,'reason':'User asked to revisit semantically implausible phrases; previous draft retained','num_beams':5,'language':'tamil','text_prompt':False,'training_admitted':False}
(OUT/'run.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor'
weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device=-1)
x,rate=sf.read(p,dtype='float32');start=time.perf_counter()
result=recognize({'raw':x,'sampling_rate':rate},return_timestamps=True,generate_kwargs={'language':'tamil','task':'transcribe','num_beams':5,'max_new_tokens':256})
status.update({'status':'completed_pending_review','elapsed_asr_seconds':time.perf_counter()-start,'result':result,'source_preserved':hashlib.sha256(p.read_bytes()).hexdigest()==before,'finished_utc':datetime.now(timezone.utc).isoformat()})
(OUT/'run.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'retry-draft.ta.txt').write_text(result['text'].strip()+'\n',encoding='utf-8')
print(json.dumps(status,ensure_ascii=True))
