"""Two requested30s source1 review pieces, separate from training inputs."""
import os,json,hashlib,time,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
CONTEXT='--context-after-165' in sys.argv
OUT=DATA/('data/context-sbs1-165-195-01' if CONTEXT else 'data/review-sbs1-30s-01');OUT.mkdir(exist_ok=False)
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
source=Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-1.mp3')
expected='7429c56bc9fcf5e2c69942e4387f8448b0bd5ed99de67638e90de104c96740f7'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
# Decode once to avoid independent MP3 seek differences between contiguous pieces.
x,sr=sf.read(source,dtype='float32');assert sr==48000 and x.ndim==1 and np.isfinite(x).all()
records=[]
for index,(start,end) in enumerate(([(165,195)] if CONTEXT else [(105,135),(135,165)]),1):
    y=x[start*sr:end*sr];assert len(y)==30*sr
    path=OUT/f'SPB1-{index:03}-source-{start:03}-{end:03}.wav'
    sf.write(path,y,sr,subtype='FLOAT')
    z,zsr=sf.read(path,dtype='float32');assert zsr==sr and np.array_equal(y,z)
    records.append({'id':f'SPB1-{index:03}','source_start_seconds':start,'source_end_seconds':end,
                    'audio':str(path),'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'speaker_review':'pending','text_review':'pending','training_admitted':False,
                    'processing':'Contiguous FLOAT crop from full decoded source; no gain/filter/speed changes',
                    'result':None})
receipt={'source':str(source),'source_sha256':expected,'source_duration_seconds':len(x)/sr,
         'intro_exclusion_seconds':[0,105],'exclusion_basis':'Explicit user instruction',
         'scope':('One30second context beyond the first minute for Claude batch drafts; no human approval implied' if CONTEXT else 'Two30second review pieces; human first batch not assumed fully aligned/approved'),
         'human_dictation_file':str(DATA/'data/human-dictation-sbs1-01/batch-001-raw-dictation.json'),
         'settings':{'resampling':'scipy resample_poly1/3','input_sample_rate':16000,'num_beams':3,
                     'text_prompt':False,'return_timestamps':True},'records':records}
def save(): (OUT/'review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
save();print(json.dumps({'audio_ready':[r['audio'] for r in records]}),flush=True)
import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.75,0)
procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor';weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device=0)
for r in records:
    y=x[r['source_start_seconds']*sr:r['source_end_seconds']*sr]
    y=resample_poly(y,1,3).astype('float32');started=time.perf_counter()
    r['result']=recognize({'raw':y,'sampling_rate':16000},return_timestamps=True,
                         generate_kwargs={'language':'tamil','task':'transcribe','num_beams':3,'max_new_tokens':440})
    r['asr_seconds']=time.perf_counter()-started;r['status']='unreviewed_machine_draft'
    save()
    body=(f"# {r['id']} — மூல நேரம் {r['source_start_seconds']}–{r['source_end_seconds']}விநாடிகள்\n\n"
          f"![30விநாடி மூல ஒலி]({Path(r['audio']).as_posix()})\n\n"
          'இது திருத்தப்படாத இயந்திர உரை; பேசுபவர்/வார்த்தைகள் இன்னும் உறுதிசெய்யப்படவில்லை.\n\n'
          +r['result']['text'].strip()+'\n')
    (OUT/(r['id']+'-DRAFT.md')).write_text(body,encoding='utf-8')
    print(json.dumps({'id':r['id'],'text':r['result']['text'],'asr_seconds':r['asr_seconds']},ensure_ascii=True),flush=True)
receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
receipt['source_preserved']=hashlib.sha256(source.read_bytes()).hexdigest()==expected
save()
