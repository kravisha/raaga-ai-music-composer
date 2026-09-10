"""Acquire pinned public training source/weights into a new isolated root."""
import hashlib,json,urllib.request,zipfile,io
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
ASSETS=ROOT/'assets'
pinned=json.loads((ASSETS/'pinned-sources.json').read_text())
e={'started_utc':datetime.now(timezone.utc).isoformat(),'files':[]}

def fetch(url,path,size=None,lfs=None,blob=None):
    if path.exists():raise FileExistsError(path)
    part=path.with_name(path.name+'.part')
    h=hashlib.sha256(); total=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Raaga-local-training-preparation/1.0'}),timeout=90) as r,part.open('xb') as f:
        while b:=r.read(1024*1024):h.update(b);f.write(b);total+=len(b)
    if size is not None and total!=size:raise ValueError('size mismatch '+str(path))
    if lfs and h.hexdigest()!=lfs:raise ValueError('sha mismatch '+str(path))
    if blob:
        gh=hashlib.sha1();gh.update(f'blob {total}\0'.encode());gh.update(part.read_bytes())
        if gh.hexdigest()!=blob:raise ValueError('git blob mismatch '+str(path))
    part.rename(path)
    row={'url':url,'path':str(path),'bytes':total,'sha256':h.hexdigest(),'upstream_digest_verified':bool(lfs or blob)}
    e['files'].append(row);print(json.dumps(row),flush=True)

revision=pinned['checkpoint']['revision']
target=ASSETS/('mms-tts-tam-train-'+revision[:12]);target.mkdir(exist_ok=False)
for f in pinned['checkpoint']['files']:
    name=f['rfilename']
    if name=='.gitattributes':continue
    fetch(f'https://huggingface.co/ylacombe/mms-tts-tam-train/resolve/{revision}/{name}',target/name,f.get('size'),f.get('lfs',{}).get('sha256'),None if 'lfs' in f else f.get('blobId'))
revision=pinned['recipe']['revision'];archive=ASSETS/f'finetune-hf-vits-{revision}.zip'
fetch(f'https://codeload.github.com/ylacombe/finetune-hf-vits/zip/{revision}',archive)
source=ROOT/'training-source';source.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        parts=Path(info.filename).parts[1:]
        if not parts:continue
        dest=source.joinpath(*parts).resolve()
        if not dest.is_relative_to(source.resolve()):raise ValueError('archive path outside destination')
        if info.is_dir():dest.mkdir(parents=True,exist_ok=True)
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
e.update({'checkpoint_revision':pinned['checkpoint']['revision'],'recipe_revision':revision,'checkpoint_dir':str(target),'training_source_dir':str(source),'finished_utc':datetime.now(timezone.utc).isoformat(),'training_started':False,'runtime_changed':False})
(ASSETS/'acquisition.json').write_text(json.dumps(e,indent=2),encoding='utf-8')
print('PINNED_ACQUISITION_COMPLETE',flush=True)
