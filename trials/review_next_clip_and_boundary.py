"""Two bounded requested ASR checks with the established16kHz frontend."""
import os,json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
OUT=DATA/'data/semantic-review-02';OUT.mkdir(exist_ok=False)
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',MPLCONFIGDIR=str(DATA/'cache/matplotlib'))
import numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly
from math import gcd
from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
torch.set_num_threads(4);torch.set_num_interop_threads(1)
procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor';weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device=-1)
for name,rel,beam,timestamps in [
    ('next-clip','data/review-batch-01/clips/cx-sbs3-060000.wav',3,False),
    ('first-boundary','data/boundary-context-01/cx-sbs2-boundary-076000.wav',1,'word')]:
    p=DATA/rel;h=hashlib.sha256(p.read_bytes()).hexdigest();x,rate=sf.read(p,dtype='float32');factor=gcd(rate,16000)
    x=resample_poly(x,16000//factor,rate//factor).astype('float32')
    start=time.perf_counter();result=recognize({'raw':x,'sampling_rate':16000},return_timestamps=timestamps,generate_kwargs={'language':'tamil','task':'transcribe','num_beams':beam,'max_new_tokens':256 if name=='next-clip' else 128})
    rec={'created_utc':datetime.now(timezone.utc).isoformat(),'source':str(p),'sha256':h,'settings':{'num_beams':beam,'timestamps':timestamps,'text_prompt':False,'input_sample_rate':16000,'resampling':'scipy resample_poly'},'seconds':time.perf_counter()-start,'result':result,'input_preserved':hashlib.sha256(p.read_bytes()).hexdigest()==h,'status':'machine_draft_only','semantic_review':'pending; not human-approved or admitted'}
    (OUT/(name+'.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'name':name,'text':result['text'],'seconds':rec['seconds']},ensure_ascii=True),flush=True)
