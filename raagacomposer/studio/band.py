"""The Agentic Band's stage (Krish, 7 Nov 2026).

"One common interface where all agents can be seated, and they play their instrument. If it is an instrumentalist, an instrument
is displayed for that agent; if it is a vocal, a mic - blue for male, pink for female." Krish is the director, Kali the lead
producer; any agent may take a seat and pick an instrument.

A seat is data: who sits there, what they play, whether they are playing. The band plays the entrance and groove from
``groove.orchestrate_show``: every instrumental part belongs to the seat that plays that instrument, a part with nobody on it is
silent, and a singer's seat sings the hook on "aa" in a male or female voice.  So the director hears exactly the seats that are
on, and an agent changing instrument changes what is heard.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..audio import dsp
from ..core.models import MelodyVersion, Note, Section, SectionKind, VocalDirection
from ..music import instruments as catalog
from ..music.synth import render_notes
from . import groove as G

VOCAL = "vocal"
INSTRUMENT = "instrument"

#: What a seat can play: the groove's parts.  A singer sings the lead line.
PLAYABLE = ("saxophone", "violin", "veena", "strings", "bass", "drum_kit", "mridangam", "kanjira", "tambourine", "synth_pad")

#: An icon for each instrument, shown on the seat.
ICONS = {"saxophone": "\U0001F3B7", "violin": "\U0001F3BB", "veena": "\U0001FA95", "strings": "\U0001F3BB", "bass": "\U0001F3B8",
         "drum_kit": "\U0001F941", "mridangam": "\U0001FA98", "kanjira": "\U0001FA98", "tambourine": "\U0001FA98",
         "synth_pad": "\U0001F3B9", VOCAL: "\U0001F3A4"}

MIC_COLOURS = {"male": "#3b82f6", "female": "#ec4899", "neutral": "#a3a3a3"}


@dataclass
class Seat:
    agent: str
    role: str = INSTRUMENT            # VOCAL or INSTRUMENT
    instrument: str = ""              # one of PLAYABLE for an instrumentalist
    voice: str = "neutral"            # male / female / neutral, for a singer's mic colour and voice
    playing: bool = True
    title: str = ""                   # e.g. "lead producer"

    @property
    def icon(self) -> str:
        return ICONS[VOCAL] if self.role == VOCAL else ICONS.get(self.instrument, "\U0001F3B5")

    @property
    def mic_colour(self) -> str:
        return MIC_COLOURS.get(self.voice, MIC_COLOURS["neutral"])

    def to_dict(self) -> dict:
        return asdict(self)


def default_band() -> list[Seat]:
    """A first seating for the director to change. Agents will pick their own seats later; these are suggestions."""
    return [
        Seat("Kali", VOCAL, voice="female", title="lead producer"),
        Seat("Bhairava", INSTRUMENT, "saxophone"),
        Seat("Loki", INSTRUMENT, "violin"),
        Seat("Mimi", INSTRUMENT, "veena"),
        Seat("Jingjing", INSTRUMENT, "drum_kit"),
        Seat("Tara", INSTRUMENT, "bass"),
        Seat("Boris", INSTRUMENT, "mridangam"),
        Seat("Diego", INSTRUMENT, "tambourine"),
        Seat("Claude Work", INSTRUMENT, "kanjira"),
        Seat("Claude Dev", INSTRUMENT, "strings"),
        Seat("Codex", INSTRUMENT, "synth_pad"),
    ]


def arrangement(seats: list[Seat], silence: float = 0.0) -> tuple[dict[str, list[Note]], list[Note], float]:
    """-> (instrument parts that someone is playing, the lead line a singer sings or [], total seconds)."""
    parts, seconds = G.orchestrate_show(silence)
    playing = [s for s in seats if s.playing]
    taken = {s.instrument for s in playing if s.role == INSTRUMENT}
    heard = {k: v for k, v in parts.items() if k in taken and v}
    lead = list(parts.get("saxophone", [])) if any(s.role == VOCAL for s in playing) else []
    return heard, lead, seconds


def render(seats: list[Seat], sr: int = 44100, silence: float = 0.0, seed: int = 7) -> np.ndarray:
    """The band as audio: only the seats that are playing."""
    from ..providers.local import LocalVoiceProvider
    from ..voice.profiles import BUILTIN

    heard, lead, seconds = arrangement(seats, silence)
    total = seconds + 1.5
    mix = dsp.silence(total, sr)
    for key, notes in heard.items():
        inst = catalog.get(key)
        if inst is None:
            raise KeyError(f"The instrument catalog has no {key}")
        gain, pan = G.MIX_PUNCH.get(key, (0.6, 0.0))
        mix = dsp.mix_into(mix, dsp.pan_mono(render_notes(notes, inst, sr, total_seconds=total, seed=seed) * gain, pan), 0)
    singers = [s for s in seats if s.playing and s.role == VOCAL]
    if lead and singers:
        voices = {p.gender: p for p in BUILTIN}
        for k, singer in enumerate(singers):
            profile = voices.get(singer.voice) or BUILTIN[0]
            shift = -12 if profile.gender == "male" else 0
            notes = [Note(swara=n.swara, midi=n.midi + shift, start=n.start, duration=n.duration, velocity=n.velocity)
                     for n in lead]
            melody = MelodyVersion(raaga=G.Groove().raaga, tonic_midi=G.TONIC + shift, tempo_bpm=G.BPM, notes=notes,
                                   sections=[Section(id="band", kind=SectionKind.PALLAVI, start=0.0, end=total)])
            sung = LocalVoiceProvider().render_vocal(melody, None, profile, VocalDirection(style="energetic"), sr, total,
                                                     seed=seed + k, vocal_sections_only=False, vowel="a")
            pan = 0.0 if len(singers) == 1 else (-0.2 + 0.4 * k / (len(singers) - 1))
            mix = dsp.mix_into(mix, dsp.pan_mono(sung * 0.85, pan), 0)
    out = dsp.normalize_loudness(mix, sr, -14.0)
    out = dsp.limiter(out, -1.0, sr)
    return dsp.fade(out, 0.0, 1.0, sr).astype(np.float32)
