import os,json
from pathlib import Path
ROOT=Path('C:/Users/Krish/ClaudeStuff');OUT=ROOT/'Raaga-SPB-Direct-TTS-20260910'
os.environ.update(HF_HOME=str(OUT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
from transformers import VitsTokenizer
model=ROOT/'raaga-wt-claude-voice-prototype/models/facebook--mms-tts-tam'
t=VitsTokenizer.from_pretrained(model,local_files_only=True)
rows=[]
for text in ['playback singer','singers','qualification','பாடக்கூடியவன்தான் ஒரு playback singer','அத']:
    ids=t(text)['input_ids'];decoded=t.decode(ids,skip_special_tokens=True)
    rows.append({'input':text,'tokens':t.convert_ids_to_tokens(ids),'decoded':decoded,'ids':ids})
record={'scope':'actual unchanged Tamil tokenizer, no TTS or training','normalization':t.normalize,'vocabulary_size':len(t.get_vocab()),'cases':rows,'decision':'Keep human verbatim transcript separate. English code-switch words require human-reviewed Tamil-script pronunciation text or a separately evaluated tokenizer expansion. Do not silently train normalized text that drops these words.'}
(OUT/'assets/tamil-token-coverage.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=True))
