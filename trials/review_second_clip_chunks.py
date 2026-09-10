"""Auditable short listening chunks and unprompted, explicitly resampled ASR."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/Krish/ClaudeStuff')
DATA = ROOT / 'Raaga-SPB-Direct-TTS-20260910'
OUT = DATA / 'data/second-clip-chunks-01'
OUT.mkdir(exist_ok=False)
os.environ.update(HF_HOME=str(DATA / 'cache/huggingface'), HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  MPLCONFIGDIR=str(DATA / 'cache/matplotlib'))
import soundfile as sf
from scipy.signal import resample_poly

source = Path('C:/Users/Krish/Documents/Sound Recordings/SPB-interview-part-3.mp3')
sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert sha == '691614afe3830562a00d1653db4847dc8555987aaba4c490989dfbc90cec5136'
windows = [(f'{n+1:02d}', 60+n*4, 64+n*4) for n in range(5)]
windows.append(('06-tail-context', 76, 84))
records = []
with sf.SoundFile(source) as stream:
    rate = stream.samplerate
    assert rate == 48000 and stream.channels == 1
    for name, start, end in windows:
        stream.seek(round(start*rate))
        x = stream.read(round((end-start)*rate), dtype='float32')
        assert len(x) == round((end-start)*rate)
        path = OUT / f'{name}-source-{start:03d}-{end:03d}.wav'
        sf.write(path, x, rate, subtype='FLOAT')
        records.append({'id':name, 'source_start_seconds':start, 'source_end_seconds':end,
                        'review_clip_start_seconds':start-60, 'audio':str(path),
                        'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'status':'listening_context_only', 'result':None})
receipt = {'source':str(source), 'source_sha256':sha, 'sample_rate':rate,
           'scope':'Context copies only, not additional corpus or approved transcripts',
           'settings':{'resampling':'scipy.signal.resample_poly 1/3', 'input_sample_rate':16000,
                       'num_beams':3, 'text_prompt':False, 'task':'transcribe', 'language':'tamil'},
           'human_confirmed_term':'சங்கதிகள்', 'chunks':records}
def save():
    (OUT / 'run.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
save()
print(json.dumps({'event':'audio_ready', 'folder':str(OUT), 'clips':[r['audio'] for r in records]}), flush=True)

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig, pipeline
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
procpath = ROOT / 'Raaga-Vocal-Quality-20260909/whisper-processor'
weights = ROOT / 'Raaga-Personal-Voice-Trial-20260908/models/openai--whisper-small'
proc = WhisperProcessor.from_pretrained(procpath, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(weights, local_files_only=True,
                         torch_dtype=torch.float32, attn_implementation='eager')
model.generation_config = GenerationConfig.from_pretrained(procpath, local_files_only=True)
recognize = pipeline('automatic-speech-recognition', model=model, tokenizer=proc.tokenizer,
                     feature_extractor=proc.feature_extractor, device=-1)
for record in records:
    x, read_rate = sf.read(record['audio'], dtype='float32')
    assert read_rate == 48000
    x = resample_poly(x, 1, 3).astype('float32')
    start = time.perf_counter()
    result = recognize({'raw':x, 'sampling_rate':16000},
                       generate_kwargs={'language':'tamil', 'task':'transcribe',
                                        'num_beams':3, 'max_new_tokens':96})
    record.update(result=result, status='machine_draft_only', asr_seconds=time.perf_counter()-start,
                  created_utc=datetime.now(timezone.utc).isoformat())
    save()
    print(json.dumps({'id':record['id'], 'text':result['text'], 'seconds':record['asr_seconds']}, ensure_ascii=True), flush=True)
receipt['source_preserved'] = hashlib.sha256(source.read_bytes()).hexdigest() == sha
receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
save()
