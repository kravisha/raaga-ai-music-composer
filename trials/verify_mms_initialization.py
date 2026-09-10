"""Compare exact saved generator tensors; never load a model or train."""
import json,struct,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff')
base=ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam'
training=ROOT/'Raaga-SPB-Direct-TTS-20260910/assets/mms-tts-tam-train-481128b97cd2'
def header(path):
    f=path.open('rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));h.pop('__metadata__',None);return f,n+8,h
a,ao,ah=header(base/'model.safetensors');b,bo,bh=header(training/'model.safetensors')
differences=[];matched=[]
for k,v in ah.items():
    if k not in bh:differences.append({'key':k,'issue':'missing'});continue
    w=bh[k]
    if v['shape']!=w['shape'] or v['dtype']!=w['dtype']:differences.append({'key':k,'issue':'shape_or_dtype'});continue
    a.seek(ao+v['data_offsets'][0]);b.seek(bo+w['data_offsets'][0])
    x=a.read(v['data_offsets'][1]-v['data_offsets'][0]);y=b.read(w['data_offsets'][1]-w['data_offsets'][0])
    if x!=y:differences.append({'key':k,'issue':'values'})
    else:matched.append(k)
extra=sorted(set(bh)-set(ah))
cfg_a=json.loads((base/'config.json').read_text());cfg_b=json.loads((training/'config.json').read_text())
config_diff={k:{'base':cfg_a.get(k),'train':cfg_b.get(k)} for k in cfg_a.keys()|cfg_b.keys() if cfg_a.get(k)!=cfg_b.get(k)}
result={'created_utc':datetime.now(timezone.utc).isoformat(),'base':str(base),'training':str(training),'matched_generator_tensor_count':len(matched),'base_tensor_count':len(ah),'differences':differences,'extra_tensor_count':len(extra),'extra_top_groups':sorted(set(k.split('.')[0] for k in extra)),'extra_non_discriminator_tensors':[k for k in extra if not k.startswith('discriminator.')],'generator_byte_identical':not differences,'config_differences':config_diff,'vocab_byte_identical':(base/'vocab.json').read_bytes()==(training/'vocab.json').read_bytes(),'training_started':False}
(ROOT/'Raaga-SPB-Direct-TTS-20260910/assets/initialization-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
