"""Boundary context only:76–84s source, with the review clip ending at80s."""
import os,json,time,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
POLY='--poly' in sys.argv
OUT=DATA/('data/boundary-context-poly-01' if POLY else 'data/boundary-context-01');OUT.mkdir(exist_ok=False)
for key,value in {'HF_HOME':str(DATA/'cache/huggingface'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','MPLCONFIGDIR':str(DATA/'cache/matplotlib')}.items():os.environ[key]=value
import numpy as np,soundfile as sf,torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
torch.set_num_threads(4);torch.set_num_interop_threads(1)
source=Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-2.mp3')
sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha=='1833b914cd5a12a997d70d9c6e74cbff4113e9161131b1d5f6de0a2a265c6066'
with sf.SoundFile(source) as f:
    rate=f.samplerate;f.seek(round(76*rate));x=f.read(round(8*rate),dtype='float32')
p=OUT/'cx-sbs2-boundary-076000.wav';sf.write(p,x,rate,subtype='FLOAT')
status={'started_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'source_sha256':sha,'start_seconds':76,'end_seconds':84,'original_review_clip_end_within_context_seconds':4,'audio':str(p),'scope':'context for possible word truncation only, not training data','num_beams':1,'timestamps':False,'text_prompt':False,'status':'transcribing'}
(OUT/'run.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor';weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device=-1)
if POLY:
    from scipy.signal import resample_poly
    from math import gcd
    factor=gcd(rate,16000);x=resample_poly(x,16000//factor,rate//factor).astype('float32');rate=16000
status['recognizer_sample_rate']=rate
status['resampling']='explicit scipy resample_poly, matching prior successful source-review path' if POLY else 'implicit ASR pipeline'
start=time.perf_counter();result=recognize({'raw':x,'sampling_rate':rate},generate_kwargs={'language':'tamil','task':'transcribe','num_beams':1,'max_new_tokens':128})
status.update({'status':'completed_unverified_draft','elapsed_asr_seconds':time.perf_counter()-start,'result':result,'source_preserved':hashlib.sha256(source.read_bytes()).hexdigest()==sha,'finished_utc':datetime.now(timezone.utc).isoformat()})
(OUT/'run.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=True))
