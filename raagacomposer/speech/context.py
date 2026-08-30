"""Conversational context manager (spec section 5.4).

Holds what "here", "there", "this part", "that instrument", "after that" and
"keep everything else" resolve to: the playhead, the selected range, the last
instrument and section mentioned, and the recent turns.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.models import ConversationTurn, Section
from .intent import Command
from .timeline_parser import TimeContext, TimeSpec

BACK_REFERENCE = re.compile(
    r"\b(that|those|it|the same|same thing|again|as before|like before|"
    r"keep everything else)\b", re.I)

# Words that point at a place in the song rather than naming one.
DEICTIC = re.compile(r"\b(here|there|this|that|now|current|these|those)\b", re.I)

# Intents that need a time range before they can act.  Removal, replacement
# and level changes act on wherever the named instrument already plays, so
# they are deliberately absent.
PLACED_INTENTS = frozenset({
    "arrange.add", "arrange.suggest", "arrange.regenerate",
    "region.lock", "region.unlock", "tune.regenerate_section",
})


@dataclass
class ConversationContext:
    playhead: float = 0.0
    selection: Optional[Tuple[float, float]] = None
    duration: float = 0.0
    sections: List[Section] = field(default_factory=list)
    last_instrument: str = ""
    last_target_instrument: str = ""
    last_section_id: str = ""
    last_track_id: str = ""
    last_region_id: str = ""
    last_time: Optional[TimeSpec] = None
    last_intent: str = ""
    last_feel_words: List[str] = field(default_factory=list)
    turns: List[ConversationTurn] = field(default_factory=list)
    listening: bool = False
    partial: str = ""

    # -- snapshots ---------------------------------------------------------
    def time_context(self) -> TimeContext:
        return TimeContext(duration=self.duration, playhead=self.playhead,
                           selection=self.selection, sections=list(self.sections))

    def section_at_playhead(self) -> Optional[Section]:
        return next((s for s in self.sections
                     if s.start <= self.playhead < s.end), None)

    # -- resolution --------------------------------------------------------
    def resolve(self, cmd: Command) -> Command:
        """Fill in whatever the creator left implicit."""
        text = cmd.text or ""

        if not cmd.instrument and BACK_REFERENCE.search(text) and self.last_instrument:
            cmd.instrument = self.last_instrument
        if cmd.intent in ("arrange.replace",) and not cmd.instrument:
            cmd.instrument = self.last_instrument

        # Only invent a time range when the creator pointed at a place.  A bare
        # "add veena" means the whole song; "add veena here" means this spot;
        # "replace violin with veena" means wherever the violin is playing.
        needs_place = cmd.intent in PLACED_INTENTS
        pointed = bool(DEICTIC.search(text))
        if needs_place and (cmd.time is None or
                            (cmd.time.start is None and cmd.time.relative is None)):
            if self.selection:
                s, e = self.selection
                cmd.time = TimeSpec(start=s, end=e, description="the selected range",
                                    source="context-selection")
            elif pointed or cmd.intent == "tune.regenerate_section":
                section = self.section_at_playhead()
                if section is not None:
                    cmd.time = TimeSpec(start=section.start, end=section.end,
                                        description=f"the {section.name}",
                                        source="context-section",
                                        section_id=section.id)
                else:
                    cmd.time = TimeSpec(start=self.playhead, end=None,
                                        description="from the playhead",
                                        source="context-playhead")

        # An open-ended range runs to the end of the containing section.
        if cmd.time and cmd.time.start is not None and cmd.time.end is None \
                and cmd.intent in PLACED_INTENTS:
            section = next((s for s in self.sections
                            if s.start <= cmd.time.start < s.end), None)
            cmd.time.end = section.end if section else self.duration
            if cmd.time.end and cmd.time.end <= cmd.time.start:
                cmd.time.end = min(self.duration, cmd.time.start + 15.0)

        if not cmd.section_id and cmd.time and cmd.time.start is not None:
            section = next((s for s in self.sections
                            if s.start <= cmd.time.start < s.end), None)
            if section:
                cmd.section_id = section.id

        if not cmd.feel_words and cmd.intent == "arrange.suggest":
            cmd.feel_words = list(self.last_feel_words)

        cmd.interpretation = cmd.interpretation or ""
        return cmd

    def remember(self, cmd: Command) -> None:
        if cmd.instrument:
            self.last_instrument = cmd.instrument
        if cmd.target_instrument:
            self.last_target_instrument = cmd.target_instrument
            self.last_instrument = cmd.target_instrument
        if cmd.section_id:
            self.last_section_id = cmd.section_id
        if cmd.time:
            self.last_time = cmd.time
        if cmd.feel_words:
            self.last_feel_words = list(cmd.feel_words)
        if cmd.known:
            self.last_intent = cmd.intent

    # -- transcript --------------------------------------------------------
    def add_turn(self, text: str, speaker: str = "creator", final: bool = True,
                 intent: str = "", interpretation: str = "",
                 status: str = "received") -> ConversationTurn:
        turn = ConversationTurn(at=time.time(), speaker=speaker, text=text,
                                final=final, intent=intent,
                                interpretation=interpretation, status=status)
        self.turns.append(turn)
        if len(self.turns) > 500:
            del self.turns[:-500]
        return turn

    def update_status(self, turn_id: str, status: str,
                      targets: Optional[List[str]] = None) -> None:
        for t in self.turns:
            if t.id == turn_id:
                t.status = status
                if targets:
                    t.targets = list(targets)
                return

    def recent(self, n: int = 12) -> List[ConversationTurn]:
        return self.turns[-n:]

    def transcript(self, n: int = 20) -> str:
        rows = []
        for t in self.recent(n):
            who = "You" if t.speaker == "creator" else "System"
            mark = "" if t.status == "received" else f"  [{t.status}]"
            rows.append(f"{who}: {t.text}{mark}")
        return "\n".join(rows)
