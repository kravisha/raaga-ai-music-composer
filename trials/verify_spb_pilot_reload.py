"""Confirm the saved pilot independently loads in the standard TTS class."""
import os,json,hashlib,sys
from pathlib import Path
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
OUT=ROOT/('pilot-fixed-text-duration-02' if '--fixed-text-duration' in sys.argv else 'pilot-one-utterance-01')
os.environ.update(HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch,numpy as np,soundfile as sf
from transformers import VitsModel,VitsTokenizer
torch.set_num_threads(4);torch.set_num_interop_threads(1)
model,info=VitsModel.from_pretrained(OUT/'checkpoint',local_files_only=True,output_loading_info=True)
assert not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs']
assert all(k.startswith('discriminator.') for k in info['unexpected_keys'])
model=model.to('cuda').eval()
tokenizer=VitsTokenizer.from_pretrained(OUT/'checkpoint',local_files_only=True)
text=json.loads((OUT/'probe-texts.json').read_text(encoding='utf-8'))['new-sentence']
torch.manual_seed(20260911)
with torch.no_grad():
    y=model(**tokenizer(text,return_tensors='pt').to('cuda')).waveform.cpu().float().numpy().reshape(-1)
reference,sr=sf.read(OUT/'adapted-new-sentence-raw.wav',dtype='float32')
assert len(y)==len(reference) and np.isfinite(y).all()
error=float(np.max(np.abs(y-reference)))
assert error<1e-4
rec={'standard_vits_model_reload_passed':True,'text':text,'duration_seconds':len(y)/sr,
     'missing_generator_keys':info['missing_keys'],'ignored_discriminator_keys':len(info['unexpected_keys']),
     'max_sample_difference_from_in_memory_render':error,'same_seed':20260911,
     'checkpoint_sha256':hashlib.sha256((OUT/'checkpoint/model.safetensors').read_bytes()).hexdigest(),
     'claim':'Saved trained checkpoint is runnable; not evidence of listening quality.'}
(OUT/'reload-verification.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(rec,ensure_ascii=True))
