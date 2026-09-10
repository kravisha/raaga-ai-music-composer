from copy import deepcopy
import pytest
from raagacomposer.core.models import Note,MelodyVersion,LyricLine,LyricsVersion
from voice_prototype.score_input import build_score_input,ScoreInputError

def fixture():
    m=MelodyVersion(version=3,tempo_bpm=120,raaga='Shankarabharanam',notes=[
        Note(swara='S',midi=60,start=.25,duration=.5),
        Note(swara='R2',midi=62.25,start=.75,duration=.25,gamaka='kampita'),
        Note(swara='R2',midi=62.25,start=1.,duration=.25)])
    l=LyricsVersion(melody_version=3,lines=[LyricLine(id='line',text='அத',syllables=['அ','த'],note_indices=[0,1,2])])
    a=[{'line_id':'line','syllable_index':0,'note_indices':[0]},
       {'line_id':'line','syllable_index':1,'note_indices':[1,2],'articulation':{'onset_beats':.1,'coda_beats':.1}}]
    return m,l,a

def test_preserves_core_metadata_microtonal_pitch_rest_tie_and_inputs():
    m,l,a=fixture(); before=deepcopy((m,l,a))
    d=build_score_input(m,l,a,ties=[[1,2]],end_beat=3)
    assert (m,l,a)==before
    assert d['events'][0]=={'kind':'rest','onset_beat':0.,'duration_beats':.5}
    assert d['events'][2]['midi']==62.25 and d['events'][2]['gamaka']=='kampita'
    assert d['events'][-1]['kind']=='rest' and d['duration_beats']==3
    assert d['alignment'][1]['available_vowel_beats']==pytest.approx(.8)
    assert d['ties']==[[1,2]] and d['raaga']==m.raaga

@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_bad_tempo_refused(value):
    m,l,a=fixture();m.tempo_bpm=value
    with pytest.raises(ScoreInputError):build_score_input(m,l,a)

def test_lyric_cannot_span_rest():
    m,l,a=fixture();m.notes[2].start=1.1
    with pytest.raises(ScoreInputError,match='rest'):build_score_input(m,l,a)

def test_unassigned_or_double_assigned_note_refused():
    m,l,a=fixture();a[1]['note_indices']=[1]
    with pytest.raises(ScoreInputError,match='linked'):build_score_input(m,l,a)
    a[1]['note_indices']=[0,1,2]
    with pytest.raises(ScoreInputError,match='already-linked'):build_score_input(m,l,a)

def test_changed_pitch_is_melisma_but_not_tie():
    m,l,a=fixture();m.notes[2].midi=64
    assert build_score_input(m,l,a)['alignment'][1]['note_indices']==[1,2]
    with pytest.raises(ScoreInputError,match='equal-pitch'):build_score_input(m,l,a,ties=[[1,2]])

def test_consonant_budget_cannot_consume_syllable():
    m,l,a=fixture();a[1]['articulation']['onset_beats']=1
    with pytest.raises(ScoreInputError,match='vowel'):build_score_input(m,l,a)

def test_wrong_version_overlap_and_note_order_refused():
    m,l,a=fixture();l.melody_version=2
    with pytest.raises(ScoreInputError,match='version'):build_score_input(m,l,a)
    l.melody_version=3;m.notes[1].start=.5
    with pytest.raises(ScoreInputError,match='nonoverlapping'):build_score_input(m,l,a)

def test_lyric_order_refused():
    m,l,a=fixture();a[0]['syllable_index']=1;a[1]['syllable_index']=0
    with pytest.raises(ScoreInputError,match='lyric order'):build_score_input(m,l,a)

def test_unsegmented_or_omitted_lyric_text_refused():
    m,l,a=fixture();l.lines.append(LyricLine(id='missing',text='நான்',syllables=[]))
    with pytest.raises(ScoreInputError,match='explicit nonempty'):build_score_input(m,l,a)
    l.lines.pop();l.lines[0].text='அத நான்'
    with pytest.raises(ScoreInputError,match='cover the lyric'):build_score_input(m,l,a)

def test_punctuation_and_spaces_survive_without_becoming_syllables():
    m,l,a=fixture();l.lines[0].text='அ த…'
    assert build_score_input(m,l,a)['lyrics'][0]['text']=='அ த…'
