"""Bounded acoustic/ASR diagnostics for the user's final-word complaint."""
import json,os,hashlib,time,sys
from pathlib import Path
ROOT=Path('C:/Users/Krish/ClaudeStuff');DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
FIXED='--fixed-text-duration' in sys.argv
PILOT=DATA/('pilot-fixed-text-duration-02' if FIXED else 'pilot-one-utterance-01');OUT=PILOT/'ending-diagnostic-01';OUT.mkdir(exist_ok=False)
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import numpy as np,soundfile as sf,torch
from scipy.signal import stft
records=[]
for label in (['adapted'] if FIXED else ['baseline','adapted']):
    path=PILOT/f'{label}-new-sentence-listen.wav'
    y,sr=sf.read(path,dtype='float32');assert sr==16000 and np.isfinite(y).all()
    width=160;env=np.sqrt(np.mean(y[:len(y)//width*width].reshape(-1,width).astype('float64')**2,axis=1))
    active=np.flatnonzero(env>max(.003,env.max()*.03))
    tail=y[-sr:]
    freq,t,z=stft(tail,fs=sr,nperseg=400,noverlap=240)
    power=np.abs(z)**2;total=float(power.sum())
    rec={'label':label,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'duration_seconds':len(y)/sr,'peak':float(np.max(np.abs(y))),'rms':float(np.sqrt(np.mean(y*y))),
         'last_20ms_rms':float(np.sqrt(np.mean(y[-320:].astype('float64')**2))),
         'last_sample':float(y[-1]),'last_active_frame_end_seconds':(int(active[-1])+1)*.01 if len(active) else None,
         'terminal_1s_highband_fraction_3khz_to_8khz':float(power[freq>=3000].sum())/total,
         'terminal_1s_spectral_centroid_hz':float((power*freq[:,None]).sum()/total),
         'warning':'Terminal1s is a fixed region, not a verified phonetic boundary. Spectral differences do not establish cause.'}
    records.append(rec)
if FIXED:
    old=DATA/'pilot-one-utterance-01/baseline-new-sentence-listen.wav'
    new=PILOT/'baseline-new-sentence-listen.wav'
    old_y,old_sr=sf.read(old,dtype='float32');new_y,new_sr=sf.read(new,dtype='float32')
    assert old_sr==new_sr and np.array_equal(old_y,new_y)
    (OUT/'baseline-reuse.json').write_text(json.dumps({'baseline_samples_identical':True,
        'previous_asr':str(DATA/'pilot-one-utterance-01/ending-diagnostic-01/baseline-asr.json'),
        'reason':'Do not rerun recognition of the identical baseline.'},indent=2),encoding='utf-8')
(OUT/'acoustic.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'acoustic':records}),flush=True)
from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
torch.set_num_threads(4);torch.set_num_interop_threads(1)
procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor';weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,feature_extractor=proc.feature_extractor,device=-1)
for rec in records:
    y,sr=sf.read(rec['path'],dtype='float32');tic=time.perf_counter()
    rec['raw_asr']=recognize({'raw':y,'sampling_rate':sr},return_timestamps='word',
                            generate_kwargs={'language':'tamil','task':'transcribe','num_beams':1,'max_new_tokens':96})
    rec['asr_seconds']=time.perf_counter()-tic
    (OUT/(rec['label']+'-asr.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'label':rec['label'],'asr':rec['raw_asr']},ensure_ascii=True),flush=True)
