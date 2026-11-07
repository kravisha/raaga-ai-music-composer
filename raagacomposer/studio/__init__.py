"""The song team: a song made by several agents under a producer's direction.

Krish's flow (7 Nov 2026), in his words: like a music producer, he first
describes a MOOD.  An agent listens and gets the mood.  Then two things
happen at the same time: one agent writes the LYRICS, another writes the
TUNE (which raaga fits; a short two-line phrase with a little alapana feel).
When both exist a third agent fits the BEAT (which tala suits this tune).
Lyrics + tune + beat is the rough picture.  The "condiments" - prelude,
fillers, bridge, a strong start and end, more instruments - come later.

This package is the SIMPLE path only.  It is built on the engines the
composer already has (raaga selection, melody, lyric fitting, tala, beat,
singing synthesis) rather than beside them, and it does not change the open
project: what it makes is a draft the producer listens to.

Design note: ``docs/SONG_TEAM.md``.
"""
from .handoff import (
    BeatProposal,
    LyricSheet,
    MoodCard,
    RoughCut,
    SongDraft,
    TeamEvent,
    TuneProposal,
)
from .team import (
    LYRICIST,
    MOOD_LISTENER,
    NOT_DONE_YET,
    PRODUCER,
    TALA_KEEPER,
    TUNESMITH,
    Lyricist,
    MoodListener,
    SongTeam,
    TalaKeeper,
    Tunesmith,
)

__all__ = [
    "LYRICIST",
    "MOOD_LISTENER",
    "NOT_DONE_YET",
    "PRODUCER",
    "TALA_KEEPER",
    "TUNESMITH",
    "BeatProposal",
    "LyricSheet",
    "Lyricist",
    "MoodCard",
    "MoodListener",
    "RoughCut",
    "SongDraft",
    "SongTeam",
    "TalaKeeper",
    "TeamEvent",
    "TuneProposal",
    "Tunesmith",
]
