"""Speak a UTF-8 Tamil text file with the saved experimental checkpoint."""
import argparse
import json
import os
from pathlib import Path
import unicodedata

ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    destination=args.output.resolve()
    if not destination.is_relative_to(ROOT.resolve()):parser.error('Output must be inside the SPB experiment folder.')
    if destination.exists():parser.error('Output already exists; choose a new filename.')
    text=args.text_file.read_text(encoding='utf-8').strip()
    if not text:parser.error('Text is empty.')
    os.environ.update(HF_HOME=str(ROOT/'cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np,soundfile as sf,torch
    from transformers import VitsModel,VitsTokenizer
    torch.set_num_threads(4);torch.manual_seed(20260911)
    checkpoint=ROOT/'pilot-one-utterance-01/checkpoint'
    tokenizer=VitsTokenizer.from_pretrained(checkpoint,local_files_only=True)
    unsupported=sorted({c for c in unicodedata.normalize('NFC',text)
                        if unicodedata.category(c)[0] in 'LM' and c not in tokenizer.get_vocab()})
    if unsupported:parser.error('Unsupported written characters; supply an explicit Tamil pronunciation representation: '+repr(unsupported))
    tokens=tokenizer(text,return_tensors='pt').to(args.device)
    if tokens.input_ids.shape[-1]>=768:parser.error('This pilot limits each request to fewer than768tokens.')
    if int(tokens.input_ids.max())>=58:parser.error('A token exceeds the unchanged model vocabulary.')
    model=VitsModel.from_pretrained(checkpoint,local_files_only=True).to(args.device).eval()
    def duration_guard(module,inputs,out):
        if not torch.isfinite(out).all() or float(out.max())>=12:
            raise ValueError('Invalid duration prediction.')
        if float(torch.ceil(torch.exp(out)).sum())*256/16000>=45:
            raise ValueError('Predicted audio exceeds the45-second pilot limit.')
    handle=model.duration_predictor.register_forward_hook(duration_guard)
    try:
        with torch.no_grad():audio=model(**tokens).waveform.cpu().float().numpy().reshape(-1)
    finally:handle.remove()
    if not len(audio) or not np.isfinite(audio).all():raise ValueError('Invalid generated audio.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects originals if another process created the target.
    with destination.open('xb') as stream:sf.write(stream,audio,16000,format='WAV',subtype='FLOAT')
    print(json.dumps({'audio':str(destination),'duration_seconds':len(audio)/16000,
                      'synthetic':True,'model':'experimental SPB one-utterance30step pilot'},ensure_ascii=True))

if __name__=='__main__':main()
