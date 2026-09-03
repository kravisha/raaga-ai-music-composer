"""Project data model.

Everything the creator can accept, lock, version or lose lives here.  The
whole tree is plain dataclasses so it round-trips through ``core.serde`` and
survives an application restart (spec section 15).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


class Stage(str, Enum):
    BRIEF = "brief"
    RAAGA = "raaga"
    TUNE = "tune"
    LYRICS = "lyrics"
    VOICE = "voice"
    ARRANGEMENT = "arrangement"
    MIX = "mix"
    EXPORT = "export"


class ApprovalState(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    LOCKED = "locked"
    REJECTED = "rejected"


# --------------------------------------------------------------------------
# Creative brief
# --------------------------------------------------------------------------
@dataclass
class CreativeBrief:
    title: str = ""
    situation: str = ""
    mood: str = "romantic"
    feel: str = ""
    language: str = "Tamil"
    song_type: str = "film song"
    duration_target: float = 150.0
    tempo_preference: Optional[int] = None
    raaga_preference: str = ""
    vocal_feel: str = ""
    instruments_preferred: List[str] = field(default_factory=list)
    instruments_avoided: List[str] = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        bits = [b for b in (self.situation, self.mood, self.feel, self.language) if b]
        return " / ".join(bits) or "(no brief yet)"


# --------------------------------------------------------------------------
# Raaga
# --------------------------------------------------------------------------
@dataclass
class RaagaChoice:
    selected: str = ""
    alternatives: List[str] = field(default_factory=list)
    rationale: str = ""
    state: ApprovalState = ApprovalState.DRAFT
    version: int = 0
    tonic_midi: int = 60
    updated_at: float = field(default_factory=now)

    @property
    def locked(self) -> bool:
        return self.state == ApprovalState.LOCKED


# --------------------------------------------------------------------------
# Melody / tune
# --------------------------------------------------------------------------
@dataclass
class Note:
    swara: str = "S"
    midi: int = 60
    start: float = 0.0
    duration: float = 0.5
    velocity: int = 90
    gamaka: str = ""
    syllable: Optional[str] = None
    section_id: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


class SectionKind(str, Enum):
    PRELUDE = "prelude"
    PALLAVI = "pallavi"
    ANUPALLAVI = "anupallavi"
    VERSE = "verse"
    CHORUS = "chorus"
    CHARANAM = "charanam"
    INTERLUDE = "interlude"
    BRIDGE = "bridge"
    OUTRO = "outro"

    @property
    def instrumental(self) -> bool:
        return self in (SectionKind.PRELUDE, SectionKind.INTERLUDE,
                        SectionKind.BRIDGE, SectionKind.OUTRO)


@dataclass
class Section:
    id: str = field(default_factory=lambda: new_id("sec_"))
    name: str = "Pallavi"
    kind: SectionKind = SectionKind.PALLAVI
    start: float = 0.0
    end: float = 8.0
    locked: bool = False
    intensity: float = 0.6

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass
class MelodyVersion:
    version: int = 1
    created_at: float = field(default_factory=now)
    label: str = ""
    raaga: str = ""
    tonic_midi: int = 60
    tempo_bpm: int = 72
    beats_per_cycle: int = 8
    notes: List[Note] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    seed: int = 0
    state: ApprovalState = ApprovalState.DRAFT
    audio_path: str = ""
    validation: List[str] = field(default_factory=list)
    parent_version: Optional[int] = None
    derived_from: str = ""

    @property
    def duration(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    def section_by_id(self, sid: str) -> Optional[Section]:
        return next((s for s in self.sections if s.id == sid), None)

    def section_at(self, t: float) -> Optional[Section]:
        return next((s for s in self.sections if s.contains(t)), None)


# --------------------------------------------------------------------------
# Lyrics
# --------------------------------------------------------------------------
@dataclass
class LyricLine:
    id: str = field(default_factory=lambda: new_id("lyr_"))
    section_id: str = ""
    text: str = ""
    syllables: List[str] = field(default_factory=list)
    note_indices: List[int] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    locked: bool = False


@dataclass
class LyricsVersion:
    version: int = 1
    created_at: float = field(default_factory=now)
    language: str = "Tamil"
    melody_version: int = 1
    lines: List[LyricLine] = field(default_factory=list)
    state: ApprovalState = ApprovalState.DRAFT
    notes: str = ""

    def line_by_id(self, lid: str) -> Optional[LyricLine]:
        return next((l for l in self.lines if l.id == lid), None)

    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


# --------------------------------------------------------------------------
# Voice
# --------------------------------------------------------------------------
@dataclass
class VoiceProfile:
    id: str = field(default_factory=lambda: new_id("voice_"))
    name: str = "Default Female"
    gender: str = "female"
    base_midi: int = 60
    range_low: int = 52
    range_high: int = 79
    formant_shift: float = 1.0
    breathiness: float = 0.12
    vibrato_rate: float = 5.4
    vibrato_depth: float = 0.16
    brightness: float = 1.0
    source_samples: List[str] = field(default_factory=list)
    builtin: bool = True
    notes: str = ""


@dataclass
class VocalDirection:
    style: str = "romantic"
    intensity: float = 0.6
    dynamics: float = 0.5
    vibrato: float = 0.5
    breath: float = 0.5
    sustain: float = 0.5
    phrase_emphasis: float = 0.5


@dataclass
class VocalRender:
    id: str = field(default_factory=lambda: new_id("take_"))
    version: int = 1
    created_at: float = field(default_factory=now)
    kind: str = "preview"
    melody_version: int = 1
    lyrics_version: int = 1
    voice_profile_id: str = ""
    direction: VocalDirection = field(default_factory=VocalDirection)
    audio_path: str = ""
    duration: float = 0.0
    state: ApprovalState = ApprovalState.DRAFT


# --------------------------------------------------------------------------
# Arrangement
# --------------------------------------------------------------------------
@dataclass
class Region:
    id: str = field(default_factory=lambda: new_id("reg_"))
    start: float = 0.0
    end: float = 0.0
    role: str = "lead"
    notes: List[Note] = field(default_factory=list)
    gain: float = 1.0
    locked: bool = False
    version: int = 1
    seed: int = 0
    audio_path: str = ""
    generated_by: str = ""
    created_at: float = field(default_factory=now)
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and start < self.end


@dataclass
class Track:
    id: str = field(default_factory=lambda: new_id("trk_"))
    instrument: str = "veena"
    display_name: str = ""
    role: str = "lead"
    regions: List[Region] = field(default_factory=list)
    mute: bool = False
    solo: bool = False
    gain: float = 1.0
    pan: float = 0.0
    locked: bool = False
    version: int = 1
    created_by: str = "creator"
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.display_name or self.instrument.replace("_", " ").title()

    @property
    def start(self) -> float:
        return min((r.start for r in self.regions), default=0.0)

    @property
    def end(self) -> float:
        return max((r.end for r in self.regions), default=0.0)

    def region_by_id(self, rid: str) -> Optional[Region]:
        return next((r for r in self.regions if r.id == rid), None)

    def regions_in(self, start: float, end: float) -> List[Region]:
        return [r for r in self.regions if r.overlaps(start, end)]


@dataclass
class ArrangementVersion:
    version: int = 1
    created_at: float = field(default_factory=now)
    label: str = ""
    tracks: List[Track] = field(default_factory=list)
    state: ApprovalState = ApprovalState.DRAFT

    def track_by_id(self, tid: str) -> Optional[Track]:
        return next((t for t in self.tracks if t.id == tid), None)

    def tracks_for_instrument(self, instrument: str) -> List[Track]:
        return [t for t in self.tracks if t.instrument == instrument]

    @property
    def any_solo(self) -> bool:
        return any(t.solo for t in self.tracks)


@dataclass
class MixVersion:
    version: int = 1
    created_at: float = field(default_factory=now)
    kind: str = "full"
    audio_path: str = ""
    duration: float = 0.0
    loudness_db: float = 0.0
    arrangement_version: int = 1
    label: str = ""


# --------------------------------------------------------------------------
# Conversation, history, jobs, errors
# --------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    id: str = field(default_factory=lambda: new_id("turn_"))
    at: float = field(default_factory=now)
    speaker: str = "creator"
    text: str = ""
    final: bool = True
    intent: str = ""
    interpretation: str = ""
    status: str = "received"
    targets: List[str] = field(default_factory=list)


@dataclass
class HistoryEntry:
    id: str = field(default_factory=lambda: new_id("hist_"))
    at: float = field(default_factory=now)
    action: str = ""
    description: str = ""
    stage: str = ""


@dataclass
class JobRecord:
    id: str = field(default_factory=lambda: new_id("job_"))
    project_id: str = ""
    job_type: str = ""
    target: str = ""
    status: str = "queued"
    progress: float = 0.0
    provider: str = "local"
    started_at: float = field(default_factory=now)
    finished_at: float = 0.0
    output: str = ""
    error: str = ""


@dataclass
class ErrorRecord:
    id: str = field(default_factory=lambda: new_id("err_"))
    at: float = field(default_factory=now)
    where: str = ""
    message: str = ""
    retries: int = 0
    fallback: str = ""


# --------------------------------------------------------------------------
# Project root
# --------------------------------------------------------------------------
@dataclass
class Project:
    project_id: str = field(default_factory=lambda: new_id("proj_"))
    title: str = "Untitled Song"
    created_at: float = field(default_factory=now)
    modified_at: float = field(default_factory=now)
    schema_version: int = 1
    current_stage: Stage = Stage.BRIEF

    brief: CreativeBrief = field(default_factory=CreativeBrief)
    raaga: RaagaChoice = field(default_factory=RaagaChoice)

    melodies: List[MelodyVersion] = field(default_factory=list)
    approved_melody: Optional[int] = None

    lyrics: List[LyricsVersion] = field(default_factory=list)
    approved_lyrics: Optional[int] = None

    voice_profile_id: str = ""
    vocal_direction: VocalDirection = field(default_factory=VocalDirection)
    vocal_renders: List[VocalRender] = field(default_factory=list)
    vocal_master_id: str = ""

    arrangements: List[ArrangementVersion] = field(default_factory=list)
    current_arrangement: Optional[int] = None

    mixes: List[MixVersion] = field(default_factory=list)

    conversation: List[ConversationTurn] = field(default_factory=list)
    history: List[HistoryEntry] = field(default_factory=list)
    jobs: List[JobRecord] = field(default_factory=list)
    errors: List[ErrorRecord] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------
    def melody(self, version: Optional[int] = None) -> Optional[MelodyVersion]:
        v = version if version is not None else self.approved_melody
        if v is None:
            return self.melodies[-1] if self.melodies else None
        return next((m for m in self.melodies if m.version == v), None)

    @property
    def locked_melody(self) -> Optional[MelodyVersion]:
        m = self.melody()
        if m and m.state in (ApprovalState.APPROVED, ApprovalState.LOCKED):
            return m
        return None

    def lyrics_version(self, version: Optional[int] = None) -> Optional[LyricsVersion]:
        v = version if version is not None else self.approved_lyrics
        if v is None:
            return self.lyrics[-1] if self.lyrics else None
        return next((l for l in self.lyrics if l.version == v), None)

    def arrangement(self, version: Optional[int] = None) -> Optional[ArrangementVersion]:
        v = version if version is not None else self.current_arrangement
        if v is None:
            return self.arrangements[-1] if self.arrangements else None
        return next((a for a in self.arrangements if a.version == v), None)

    def vocal_render(self, render_id: str) -> Optional[VocalRender]:
        return next((r for r in self.vocal_renders if r.id == render_id), None)

    @property
    def vocal_master(self) -> Optional[VocalRender]:
        if self.vocal_master_id:
            found = self.vocal_render(self.vocal_master_id)
            if found:
                return found
        masters = [r for r in self.vocal_renders if r.kind == "master"]
        return masters[-1] if masters else None

    @property
    def latest_vocal(self) -> Optional[VocalRender]:
        return self.vocal_renders[-1] if self.vocal_renders else None

    def latest_mix(self, kind: str = "full") -> Optional[MixVersion]:
        items = [m for m in self.mixes if m.kind == kind]
        return items[-1] if items else None

    @property
    def sections(self) -> List[Section]:
        m = self.melody()
        return m.sections if m else []

    @property
    def duration(self) -> float:
        candidates: List[float] = []
        m = self.melody()
        if m:
            candidates.append(m.duration)
        arr = self.arrangement()
        if arr:
            candidates.extend(t.end for t in arr.tracks)
        for mix in self.mixes:
            candidates.append(mix.duration)
        for take in self.vocal_renders:
            candidates.append(take.duration)
        usable = [c for c in candidates if c and c > 0]
        return max(usable) if usable else float(self.brief.duration_target)

    def log_history(self, action: str, description: str) -> HistoryEntry:
        entry = HistoryEntry(action=action, description=description,
                             stage=self.current_stage.value)
        self.history.append(entry)
        if len(self.history) > 2000:
            del self.history[:-2000]
        return entry

    def touch(self) -> None:
        self.modified_at = now()
