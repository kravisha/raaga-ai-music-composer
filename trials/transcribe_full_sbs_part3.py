"""Whole-source machine draft, streamed in order; no speaker admission or clips."""
import hashlib,json,os,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff')
DATA=ROOT/'Raaga-SPB-Direct-TTS-20260910'
OUT=DATA/'data/full-transcript-sbs3-01'
SOURCE=Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-3.mp3')
EXPECTED='691614afe3830562a00d1653db4847dc8555987aaba4c490989dfbc90cec5136'
os.environ.update(HF_HOME=str(DATA/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',MPLCONFIGDIR=str(DATA/'cache/matplotlib'))

def stamp(seconds,srt=False):
    ms=round(seconds*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}{"," if srt else "."}{ms:03}'

if '--worker' not in sys.argv:
    OUT.mkdir(exist_ok=False);(OUT/'sections').mkdir();(OUT/'raw').mkdir()
    (OUT/'transcribe-script.py').write_bytes(Path(__file__).read_bytes())
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'timeout_seconds':1800,
             'source':str(SOURCE),'scope':'Entire source, all speakers/music retained for review; no training admission'}
    with (OUT/'worker.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-B',__file__,'--worker'],stdout=log,stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['pid']=p.pid;(OUT/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt),flush=True)
        try:receipt['exit_code']=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:p.kill();p.wait();receipt.update(exit_code=p.returncode,timeout=True)
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True);sys.exit(0 if receipt['exit_code']==0 else 1)

status={'stage':'loading','completed_windows':0,'covered_seconds':0,'source':str(SOURCE),
        'source_sha256':EXPECTED,'transcript_status':'raw_machine_draft; human corrections separate',
        'speaker_labels':'unverified; includes interviewer, SPB, announcements, music as present',
        'window_timing':'Exact source windows, not forced alignment or acoustic word certainty',
        'settings':{'model':'local openai/whisper-small','dtype':'float32','device':'cuda',
                    'resampling':'explicit scipy resample_poly 1/3','sampling_rate':16000,
                    'window_seconds':30,'num_beams':3,'text_prompt':False,'return_timestamps':True},
        'windows':[]}
def save(**kw):
    status.update(kw);(OUT/'coverage.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(kw,ensure_ascii=True),flush=True)

try:
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from collections import Counter
    from transformers import WhisperProcessor,WhisperForConditionalGeneration,GenerationConfig,pipeline
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
    x,rate=sf.read(SOURCE,dtype='float32')
    assert rate==48000 and x.ndim==1 and np.isfinite(x).all()
    duration=len(x)/rate
    audio=resample_poly(x,1,3).astype('float32');del x
    count=int(np.ceil(duration/30))
    procpath=ROOT/'Raaga-Vocal-Quality-20260909/whisper-processor'
    weights=ROOT/'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.75,0)
    proc=WhisperProcessor.from_pretrained(procpath,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(weights,local_files_only=True,
                                 torch_dtype=torch.float32,attn_implementation='eager')
    model.generation_config=GenerationConfig.from_pretrained(procpath,local_files_only=True)
    recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=proc.tokenizer,
                       feature_extractor=proc.feature_extractor,device=0)
    header=(f'# எஸ்.பி.பி. பேட்டி — மூன்றாம் பாகத்தின் முழு தானியங்கி உரை\n\n'
            f'முழுப் பதிவு: {stamp(duration)}. [மூல ஒலி]({SOURCE.as_posix()}).\n\n'
            'இது திருத்தப்படாத இயந்திர வரைவு. பேசுபவர் அடையாளம் உறுதி செய்யப்படவில்லை; '
            'பேட்டி எடுப்பவர், அறிவிப்புகள், இசை உள்ளிட்ட பகுதிகள் நீக்கப்படவில்லை. '
            'நேரங்கள் மூலப் பதிவின்30விநாடிச் சாளரங்களைக் குறிக்கும்; சொல் அளவிலான சீரமைப்பு அல்ல. '
            'சாளர எல்லையில் சொல் வெட்டப்படலாம். மனிதர் உறுதி செய்யாததை பயிற்சிக்கு பயன்படுத்த வேண்டாம்.\n\n')
    (OUT/'FULL-TRANSCRIPT-DRAFT.md').write_text(header,encoding='utf-8')
    (OUT/'FULL-TRANSCRIPT-DRAFT.txt').write_text(header,encoding='utf-8')
    (OUT/'FULL-TRANSCRIPT-DRAFT.srt').write_text('',encoding='utf-8')
    save(stage='transcribing',duration_seconds=duration,total_windows=count,
         audio_coverage_rule='Every contiguous source interval processed, even silence or unintelligible speech.')
    started=time.perf_counter()
    for i in range(count):
        start=i*30.;end=min((i+1)*30.,duration);ident=f'SPB3-{i+1:03}'
        chunk=audio[round(start*16000):round(end*16000)]
        tic=time.perf_counter()
        result=recognize({'raw':chunk,'sampling_rate':16000},return_timestamps=True,
                         generate_kwargs={'language':'tamil','task':'transcribe','num_beams':3,'max_new_tokens':440})
        text=result.get('text','').strip()
        words=text.split();grams=Counter(tuple(words[j:j+4]) for j in range(max(0,len(words)-3)))
        flags=[]
        if not text:flags.append('no_machine_text')
        if grams and max(grams.values())>=3:flags.append('repetition_or_hallucination_suspected; do_not_treat_as_verbatim')
        rec={'id':ident,'source_start_seconds':start,'source_end_seconds':end,'speaker':'unverified',
             'text':text,'flags':flags,'result':result,'asr_seconds':time.perf_counter()-tic,
             'status':'machine_draft_only','source_sha256':EXPECTED,'text_prompt':False}
        (OUT/'raw'/f'{ident}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        shown=text or '[தானியங்கி உரை கிடைக்கவில்லை; ஒலியைச் சரிபார்க்கவும்.]'
        block=f'## {ident} | {stamp(start)} – {stamp(end)}\n\nபேசுபவர்: உறுதி செய்யப்படவில்லை.\n\n{shown}\n\n'
        if flags:block+='குறிப்பு: '+', '.join(flags)+'\n\n'
        (OUT/'sections'/f'{ident}.md').write_text(block,encoding='utf-8')
        with (OUT/'FULL-TRANSCRIPT-DRAFT.md').open('a',encoding='utf-8') as f:f.write(block)
        with (OUT/'FULL-TRANSCRIPT-DRAFT.txt').open('a',encoding='utf-8') as f:f.write(block)
        with (OUT/'FULL-TRANSCRIPT-DRAFT.srt').open('a',encoding='utf-8') as f:
            f.write(f'{i+1}\n{stamp(start,True)} --> {stamp(end,True)}\n[{ident}; பேசுபவர் உறுதியாகவில்லை] {shown}\n\n')
        status['windows'].append({k:rec[k] for k in ['id','source_start_seconds','source_end_seconds','speaker','text','flags','asr_seconds','status']})
        elapsed=time.perf_counter()-started
        save(completed_windows=i+1,covered_seconds=end,elapsed_asr_seconds=elapsed,
             estimated_remaining_seconds=elapsed/(i+1)*(count-i-1))
    assert abs(status['covered_seconds']-duration)<1e-6
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
    save(stage='completed',source_preserved=True,finished_utc=datetime.now(timezone.utc).isoformat(),
         warning='Complete time coverage is not transcript correctness or verified speaker attribution.')
    (OUT/'FULL-TRANSCRIPT-DRAFT.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
except Exception as ex:
    import traceback
    save(stage='failed',error_type=type(ex).__name__,error=str(ex));traceback.print_exc();sys.exit(1)
