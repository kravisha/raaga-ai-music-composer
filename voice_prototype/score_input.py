"""Explicit lyric links over Raaga's existing Note/MelodyVersion structures.

This serializable boundary preserves score events beside audio. It does not
infer syllables or notes from a WAV, or encode notation as ordinary TTS text.
"""
from __future__ import annotations

import copy
import math
import unicodedata
from dataclasses import asdict

from raagacomposer.core.models import MelodyVersion, LyricsVersion


class ScoreInputError(ValueError):
    pass


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScoreInputError(f'{name} must be finite')
    if value < 0 or (positive and value == 0):
        raise ScoreInputError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def build_score_input(melody: MelodyVersion, lyrics: LyricsVersion,
                      alignment: list[dict], *, ties=(), end_beat=None) -> dict:
    """Export beat-based controls without introducing a second melody model.

    `alignment` items name line_id, syllable_index and ordered note_indices.
    Optional articulation fields onset_beats/coda_beats reserve edge budgets;
    the remaining duration is available for vowel sustain. Optional phonemes
    are carried as annotations, never guessed. Ties join equal-pitch adjacent
    notes under the same syllable; a melisma may change pitch.
    """
    tempo = _number(melody.tempo_bpm, 'tempo_bpm', positive=True)
    if lyrics.melody_version != melody.version:
        raise ScoreInputError('lyrics refer to another melody version')
    if not melody.notes:
        raise ScoreInputError('no melody notes')
    seconds_per_beat = 60.0 / tempo
    lines = {line.id: line for line in lyrics.lines}
    if len(lines) != len(lyrics.lines):
        raise ScoreInputError('duplicate lyric line id')
    def written_tokens(text):
        return ''.join(c for c in unicodedata.normalize('NFC',text)
                       if not c.isspace() and not unicodedata.category(c).startswith('P'))
    for line in lyrics.lines:
        if not isinstance(line.text,str) or not line.text.strip() or not line.syllables or any(not isinstance(s,str) or not s.strip() for s in line.syllables):
            raise ScoreInputError('lyric line needs text and explicit nonempty syllables')
        if written_tokens(line.text) != written_tokens(''.join(line.syllables)):
            raise ScoreInputError('written syllables do not cover the lyric text')
    notes, events, previous_end = [], [], 0.0
    for i, n in enumerate(melody.notes):
        start = _number(n.start, f'note {i} start') / seconds_per_beat
        duration = _number(n.duration, f'note {i} duration', positive=True) / seconds_per_beat
        pitch = _number(n.midi, f'note {i} pitch', positive=True)
        if pitch >= 128:
            raise ScoreInputError(f'note {i} pitch outside MIDI range')
        if start < previous_end - 1e-9:
            raise ScoreInputError('voice notes must be ordered and nonoverlapping')
        if start > previous_end + 1e-9:
            events.append({'kind':'rest','onset_beat':previous_end,'duration_beats':start-previous_end})
        event = {'kind':'note','note_index':i,'onset_beat':start,
                 'duration_beats':duration,'midi':n.midi,'swara':n.swara,
                 'gamaka':n.gamaka,'velocity':n.velocity,'section_id':n.section_id}
        notes.append(event); events.append(event); previous_end = start + duration
    end = previous_end if end_beat is None else _number(end_beat, 'end_beat', positive=True)
    if end < previous_end - 1e-9:
        raise ScoreInputError('end_beat cuts off a note')
    if end > previous_end + 1e-9:
        events.append({'kind':'rest','onset_beat':previous_end,'duration_beats':end-previous_end})

    mapped, used_syllables, links = {}, set(), []
    for raw in alignment:
        line = lines.get(raw.get('line_id'))
        si = raw.get('syllable_index')
        if line is None or isinstance(si,bool) or not isinstance(si,int) or not 0 <= si < len(line.syllables):
            raise ScoreInputError('unknown lyric line or syllable index')
        key = (line.id,si)
        if key in used_syllables:
            raise ScoreInputError('syllable mapped twice')
        syllable = line.syllables[si]
        if not isinstance(syllable,str) or not syllable.strip():
            raise ScoreInputError('empty syllable')
        indexes = raw.get('note_indices')
        if not isinstance(indexes,list) or not indexes:
            raise ScoreInputError('syllable needs note links')
        for ni in indexes:
            if isinstance(ni,bool) or not isinstance(ni,int) or not 0 <= ni < len(notes) or ni in mapped:
                raise ScoreInputError('invalid or already-linked note index')
            if line.note_indices and ni not in line.note_indices:
                raise ScoreInputError('link leaves the lyric line note selection')
            mapped[ni] = key
        for a,b in zip(indexes,indexes[1:]):
            if b != a+1 or abs(notes[a]['onset_beat']+notes[a]['duration_beats']-notes[b]['onset_beat'])>1e-9:
                raise ScoreInputError('a syllable cannot cross a rest or skip/reorder notes')
        span = sum(notes[i]['duration_beats'] for i in indexes)
        articulation = copy.deepcopy(raw.get('articulation',{}))
        if not isinstance(articulation,dict):
            raise ScoreInputError('articulation must be an object')
        onset = _number(articulation.get('onset_beats',0),'onset_beats')
        coda = _number(articulation.get('coda_beats',0),'coda_beats')
        if onset+coda >= span:
            raise ScoreInputError('consonant budgets leave no vowel sustain')
        used_syllables.add(key)
        links.append({'line_id':line.id,'syllable_index':si,'text':syllable,
                      'note_indices':list(indexes),'onset_beat':notes[indexes[0]]['onset_beat'],
                      'duration_beats':span,'articulation':articulation,
                      'available_vowel_beats':span-onset-coda})
    expected={(line.id,i) for line in lyrics.lines for i in range(len(line.syllables))}
    if used_syllables != expected or len(mapped)!=len(notes):
        raise ScoreInputError('every supplied syllable and voice note must be linked')
    ordered = sorted(links,key=lambda x:x['onset_beat'])
    order={line.id:i for i,line in enumerate(lyrics.lines)}
    if [(order[x['line_id']],x['syllable_index']) for x in ordered] != sorted((order[x['line_id']],x['syllable_index']) for x in links):
        raise ScoreInputError('score reverses the supplied lyric order')
    clean_ties=[]
    for pair in ties:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:
            raise ScoreInputError('tie must name two note indices')
        a,b=pair
        if any(isinstance(i,bool) or not isinstance(i,int) or i not in mapped for i in pair):
            raise ScoreInputError('tie references unknown note')
        if b!=a+1 or notes[a]['midi']!=notes[b]['midi'] or mapped[a]!=mapped[b] or abs(notes[a]['onset_beat']+notes[a]['duration_beats']-notes[b]['onset_beat'])>1e-9:
            raise ScoreInputError('tie requires adjacent equal-pitch notes in one syllable')
        if [a,b] in clean_ties:raise ScoreInputError('duplicate tie')
        clean_ties.append([a,b])
    return {'schema_version':1,'source':'Raaga MelodyVersion and LyricsVersion',
            'melody_version':melody.version,'lyrics_version':lyrics.version,
            'tempo_bpm':tempo,'beat_unit':'quarter_note','beats_per_cycle':melody.beats_per_cycle,
            'raaga':melody.raaga,'tonic_midi':melody.tonic_midi,'duration_beats':end,
            'events':events,'lyrics':[asdict(line) for line in lyrics.lines],
            'alignment':links,'ties':clean_ties,'audio_path':melody.audio_path,
            'render_status':'structured_controls_only'}
