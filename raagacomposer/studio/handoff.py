r"""What the song team's agents hand to each other.

Each agent writes one card and reads only the cards before it, so an agent
can be replaced - by a local model, by Claude, by another machine on the
team board - as long as it reads and writes the same card.  Every card has
``to_dict`` so it can travel as JSON; the heavy musical objects (notes, the
melody) travel as plain lists of swaras, pitches and times.

    MoodCard  --->  LyricSheet  (Lyricist)      \
              \->   TuneProposal (Tunesmith)     >-> BeatProposal (Tala keeper)
                                                /         |
                    LyricSheet + TuneProposal + BeatProposal -> RoughCut
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.models import BeatVersion, CreativeBrief, LyricsVersion, MelodyVersion, Note


def _notes(notes: list[Note]) -> list[dict[str, Any]]:
    return [{"swara": n.swara, "midi": n.midi, "start": round(n.start, 3),
             "duration": round(n.duration, 3),
             **({"syllable": n.syllable} if n.syllable else {})}
            for n in notes]


@dataclass
class TeamEvent:
    """One line of the team's log: who did what, and who it went to."""

    agent: str
    said: str
    handed_to: str = ""
    at: float = field(default_factory=time.time)

    def line(self) -> str:
        to = f"  ->  {self.handed_to}" if self.handed_to else ""
        return f"{self.agent}: {self.said}{to}"


@dataclass
class MoodCard:
    """What the mood listener heard.  Read by everyone after it."""

    text: str
    language: str
    feel_words: list[str]
    themes: list[str]
    tempo_bpm: int
    pace: str
    #: The one agreement the lyricist and the tunesmith share before either
    #: starts: how many lines, and roughly how many syllables each.  They
    #: work at the same time, so neither can wait for the other's answer.
    line_plan: list[int]
    brief: CreativeBrief = field(repr=False, default_factory=CreativeBrief)

    def summary(self) -> str:
        words = ", ".join(self.feel_words[:6]) or "nothing I recognise yet"
        return (f"I hear: {words}. Pace {self.pace} ({self.tempo_bpm} bpm). "
                f"{len(self.line_plan)} lines of about "
                f"{self.line_plan[0] if self.line_plan else 0} syllables, "
                f"in {self.language}.")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "language": self.language,
                "feel_words": list(self.feel_words), "themes": list(self.themes),
                "tempo_bpm": self.tempo_bpm, "pace": self.pace,
                "line_plan": list(self.line_plan)}


@dataclass
class LyricSheet:
    """What the lyricist wrote, from the mood alone."""

    lines: list[str]
    language: str
    written_by: str                  # "lexicon engine" or the model's name
    #: After the fit: the words as they are sung, phrase by phrase.
    fitted: LyricsVersion | None = field(default=None, repr=False)
    revised: list[int] = field(default_factory=list)
    fit_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"lines": list(self.lines),
                               "language": self.language,
                               "written_by": self.written_by,
                               "revised_lines": list(self.revised),
                               "fit_notes": list(self.fit_notes)}
        if self.fitted is not None:
            out["sung_phrases"] = [
                {"text": l.text, "syllables": list(l.syllables),
                 "start": round(l.start, 3), "end": round(l.end, 3)}
                for l in self.fitted.lines]
        return out


@dataclass
class TuneProposal:
    """What the tunesmith proposes: a raaga, why, and a short phrase in it."""

    raaga: str
    reason: str
    chosen_by: str                   # "tunesmith" or "producer"
    alternatives: list[tuple[str, str]]
    arohanam: list[str]
    avarohanam: list[str]
    alapana: list[Note]
    melody: MelodyVersion = field(repr=False, default_factory=MelodyVersion)
    #: The phrase's swaras, one string per lyric line.
    phrase_lines: list[str] = field(default_factory=list)
    seed: int = 0

    @property
    def tonic_midi(self) -> int:
        return self.melody.tonic_midi

    @property
    def tempo_bpm(self) -> int:
        return self.melody.tempo_bpm

    @property
    def alapana_seconds(self) -> float:
        return max((n.end for n in self.alapana), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"raaga": self.raaga, "reason": self.reason,
                "chosen_by": self.chosen_by,
                "alternatives": [list(a) for a in self.alternatives],
                "arohanam": list(self.arohanam),
                "avarohanam": list(self.avarohanam),
                "tonic_midi": self.tonic_midi, "tempo_bpm": self.tempo_bpm,
                "beats_per_cycle": self.melody.beats_per_cycle,
                "seed": self.seed, "phrase_lines": list(self.phrase_lines),
                "alapana": _notes(self.alapana),
                "phrase": _notes(self.melody.notes)}


@dataclass
class BeatProposal:
    """What the tala keeper proposes for this tune and these words."""

    tala: str
    description: str
    reason: str
    chosen_by: str                   # "tala keeper" or "producer"
    alternatives: list[tuple[str, str]]
    tempo_bpm: int
    density: str
    beat: BeatVersion = field(repr=False, default_factory=BeatVersion)
    #: True when the tune had to be re-set to this cycle.
    recast_tune: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"tala": self.tala, "description": self.description,
                "reason": self.reason, "chosen_by": self.chosen_by,
                "alternatives": [list(a) for a in self.alternatives],
                "tempo_bpm": self.tempo_bpm, "density": self.density,
                "recast_tune": self.recast_tune,
                "strokes": len(self.beat.notes)}


@dataclass
class RoughCut:
    """The rough picture as sound, and plainly what is and is not in it."""

    path: str
    seconds: float
    layers: list[str]
    missing: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "seconds": round(self.seconds, 2),
                "layers": list(self.layers), "missing": list(self.missing)}


@dataclass
class SongDraft:
    """Everything one run of the team produced."""

    mood: MoodCard
    lyrics: LyricSheet
    tune: TuneProposal
    beat: BeatProposal
    events: list[TeamEvent] = field(default_factory=list)
    not_done: list[str] = field(default_factory=list)
    rough: RoughCut | None = None

    @property
    def song_seconds(self) -> float:
        """Alapana, then the two lines."""
        return self.tune.alapana_seconds + self.tune.melody.duration

    def to_dict(self) -> dict[str, Any]:
        return {"mood": self.mood.to_dict(), "lyrics": self.lyrics.to_dict(),
                "tune": self.tune.to_dict(), "beat": self.beat.to_dict(),
                "events": [e.line() for e in self.events],
                "not_done": list(self.not_done),
                "rough": self.rough.to_dict() if self.rough else None}
