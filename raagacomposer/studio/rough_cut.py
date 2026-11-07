"""The rough picture, as sound: what the existing audio pipeline can play.

Layers, in order of time:

* a tanpura under everything, plucked P-S-S-S the way a tanpura is;
* the alapana sketch, sung on "aa" (akaram), unmetered;
* the two lines, sung with the lyricist's words fitted to the phrase;
* the mridangam in the chosen tala, under the two lines only - an alapana
  has no beat.

All of it goes through the providers the composer already uses
(``LocalMusicProvider``, ``LocalVoiceProvider`` by default), so the rough
cut sounds exactly as good as the composer does and no better.  What it
cannot make is listed on the cut itself (``RoughCut.missing``).
"""
from __future__ import annotations

import numpy as np

from ..audio import dsp
from ..core.models import (
    MelodyVersion,
    Note,
    Section,
    SectionKind,
    VocalDirection,
    VoiceProfile,
)
from ..voice import mastering
from .handoff import SongDraft

#: Seconds of silence after the alapana before the first line.
BREATH = 0.6
#: Seconds of tail after the last line, so the tanpura can ring out.
TAIL = 1.8


def _style_for(pace: str, feel_words: list[str]) -> str:
    words = set(feel_words)
    if words & {"devotional", "prayerful"}:
        return "devotional"
    if words & {"sad", "grief", "lonely", "pathos"}:
        return "sad"
    if pace == "lively":
        return "energetic"
    return "romantic"


def tanpura_notes(tonic: int, seconds: float) -> list[Note]:
    """P - S+ - S+ - S, repeated: the drone the raaga sits on."""
    out: list[Note] = []
    pluck = 0.42
    pattern = (("P", tonic - 17), ("S", tonic), ("S", tonic), ("S", tonic - 12))
    t = 0.0
    while t < seconds - 0.1:
        for swara, midi in pattern:
            if t >= seconds - 0.1:
                break
            out.append(Note(swara=swara, midi=midi, start=round(t, 3),
                            duration=1.6, velocity=46))
            t += pluck
        t += 0.25
    return out


def _shifted(notes: list[Note], by: float) -> list[Note]:
    return [Note(swara=n.swara, midi=n.midi, start=round(n.start + by, 4),
                 duration=n.duration, velocity=n.velocity, gamaka=n.gamaka,
                 syllable=n.syllable, section_id=n.section_id)
            for n in notes]


def render(draft: SongDraft, sr: int, *, music=None, voice=None,
           profile: VoiceProfile | None = None, seed: int = 0
           ) -> tuple[np.ndarray, list[str]]:
    """Mix the rough picture.  Returns stereo audio and the layers in it."""
    from ..providers.local import LocalMusicProvider, LocalVoiceProvider
    from ..voice.profiles import BUILTIN

    music = music or LocalMusicProvider()
    voice = voice or LocalVoiceProvider()
    profile = profile or BUILTIN[0]
    tune = draft.tune
    offset = tune.alapana_seconds + BREATH
    total = offset + tune.melody.duration + TAIL
    direction = VocalDirection(style=_style_for(draft.mood.pace,
                                                draft.mood.feel_words))
    layers: list[str] = []
    mix = dsp.silence(total, sr)

    # The alapana, sung on "aa" with no words and no metre.
    alapana = MelodyVersion(
        raaga=tune.raaga, tonic_midi=tune.tonic_midi,
        tempo_bpm=tune.tempo_bpm, beats_per_cycle=tune.melody.beats_per_cycle,
        notes=list(tune.alapana),
        sections=[Section(id="alapana", name="Alapana",
                          kind=SectionKind.PALLAVI, start=0.0,
                          end=tune.alapana_seconds)])
    sung_alapana = voice.render_vocal(alapana, None, profile, direction, sr,
                                      tune.alapana_seconds + 0.6, seed=seed,
                                      vocal_sections_only=False, vowel="a")
    mix = dsp.mix_into(mix, mastering.quick_preview(sung_alapana, sr, direction), 0)
    layers.append(f"alapana sketch in {tune.raaga}, sung on 'aa' "
                  f"({tune.alapana_seconds:.1f}s)")

    # The two lines, with the words.
    lines = MelodyVersion(
        raaga=tune.raaga, tonic_midi=tune.tonic_midi,
        tempo_bpm=tune.tempo_bpm, beats_per_cycle=tune.melody.beats_per_cycle,
        notes=_shifted(tune.melody.notes, offset),
        sections=[Section(id=s.id, name=s.name, kind=s.kind,
                          start=s.start + offset, end=s.end + offset)
                  for s in tune.melody.sections])
    fitted = draft.lyrics.fitted
    if fitted is not None:
        for line in fitted.lines:
            line.start += offset
            line.end += offset
    try:
        sung = voice.render_vocal(lines, fitted, profile, direction, sr,
                                  total, seed=seed + 1)
    finally:
        if fitted is not None:            # the card keeps its own times
            for line in fitted.lines:
                line.start -= offset
                line.end -= offset
    mix = dsp.mix_into(mix, mastering.quick_preview(sung, sr, direction), 0)
    layers.append(f"two lines sung with the words, {tune.tempo_bpm} bpm")

    # Mridangam under the lines only.
    strokes = _shifted(draft.beat.beat.notes, offset)
    if strokes:
        beat = music.render_part(strokes, "mridangam", sr,
                                 total_seconds=total, seed=seed + 2)
        mix = dsp.mix_into(mix, dsp.pan_mono(beat * 0.55, 0.05), 0)
        layers.append(f"mridangam in {draft.beat.tala} ({draft.beat.density})")

    # The drone, quietly, under everything.
    drone = music.render_part(tanpura_notes(tune.tonic_midi, total - 0.5),
                              "tanpura", sr, total_seconds=total, seed=seed + 3)
    mix = dsp.mix_into(mix, dsp.pan_mono(drone * 0.35, -0.1), 0)
    layers.append("tanpura drone")

    out = dsp.normalize_loudness(dsp.as_stereo(mix), sr, -16.0)
    out = dsp.limiter(out, -1.0, sr)
    out = dsp.fade(out, 0.05, 1.2, sr)
    return out.astype(np.float32), layers
