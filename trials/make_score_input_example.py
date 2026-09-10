"""Illustrative authored score, not a transcription of an SPB performance."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raagacomposer.core.models import Note,MelodyVersion,LyricLine,LyricsVersion
from voice_prototype.score_input import build_score_input

out=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
tempo=120
durations=[.5,.5,1.,.5,.5,.25,.25,1.,.5]
pitches=[60,62,64,65,64,62,64,60,60]
swaras=['S','R2','G3','M1','G3','R2','G3','S','S']
notes=[];beat=.5
for d,p,s in zip(durations,pitches,swaras):
    notes.append(Note(swara=s,midi=p,start=beat*.5,duration=d*.5,section_id='example-pallavi'))
    beat+=d
m=MelodyVersion(version=1,label='Illustrative lyric score',raaga='Shankarabharanam',tonic_midi=60,tempo_bpm=tempo,notes=notes)
line=LyricLine(id='example-line',section_id='example-pallavi',text='அத நான் கத்துக்கிட்டு',syllables=['அ','த','நான்','கத்','துக்','கிட்','டு'],note_indices=list(range(9)))
l=LyricsVersion(melody_version=1,lines=[line])
mapping=[[0],[1],[2,3],[4],[5],[6],[7,8]]
links=[{'line_id':line.id,'syllable_index':i,'note_indices':indices,'articulation':{'onset_beats':.04,'coda_beats':.02}} for i,indices in enumerate(mapping)]
score=build_score_input(m,l,links,ties=[[7,8]],end_beat=beat+.5)
score['example_notice']='Authored structural illustration; pitches, timing and syllable segmentation are examples, not approved singing or a measured SPB score.'
(out/'score-input-example.json').write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(out/'score-input-example.json'),'tempo':tempo,'duration_beats':score['duration_beats'],'syllables':len(links),'notes':len(notes),'rests':sum(e['kind']=='rest' for e in score['events'])}))
