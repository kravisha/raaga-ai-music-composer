"""Phases B and C - getting the content, and getting it ready.

Specification section 7.  Acquisition only ever uses the representation
:mod:`access` already decided is permitted, so there is no second place where
a rule about what may be fetched has to be remembered.  Three shapes arrive
here and no others:

``exercise``    notes the application renders from its own raaga library and
                then performs, so it can listen to its own playing.
``audio``       a file on disk the creator supplied.
``transcript``  text the creator supplied, or captions they pasted in.

Preparation differs by shape.  Audio from a real recording goes through
:mod:`raagacomposer.agent.preprocess` first - the drone is found and its Sa
believed, and the stretches that are talking rather than singing are silenced
- because a lesson recording is a person speaking over a tanpura and the ears
were built for one clean voice.  A rendered exercise skips all of that: it is
already one clean voice, and preparing it could only take good phrases away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agent import analysis, preprocess
from ..core.logging_setup import get_logger
from ..core.models import Note
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import Raaga, RaagaLibrary
from .access import AccessDecision
from .models import LearningSource

log = get_logger("training.acquisition")

ANALYSIS_SR = 22050
DEFAULT_TONIC_MIDI = 60


@dataclass
class AcquiredContent:
    """What we managed to get, and what it cost to get it."""

    representation: str = "none"          # exercise | audio | transcript
    audio: Optional[np.ndarray] = None
    sample_rate: int = ANALYSIS_SR
    transcript: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    raaga: Optional[Raaga] = None
    tonic_midi: Optional[float] = None
    expected_swaras: List[str] = field(default_factory=list)
    prepared: Optional[preprocess.PreparedAudio] = None
    language: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.representation != "none"

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0

    def describe(self) -> str:
        if self.representation == "transcript":
            return f"transcript of {len(self.transcript.split())} word(s)"
        if self.has_audio:
            seconds = len(self.audio) / max(1, self.sample_rate)
            text = f"{self.representation} audio, {seconds:.0f}s"
            if self.prepared is not None and self.prepared.silenced_seconds:
                text += (f" ({self.prepared.silenced_seconds:.0f}s silenced "
                         f"as speech)")
            return text
        return self.representation


# --------------------------------------------------------------------------
# exercises the application performs for itself
# --------------------------------------------------------------------------
#: What each Training-tab topic actually asks the agent to play.  Returning
#: the swaras as well as the audio is what lets the analysis be *marked*: we
#: know what was played, so we can ask whether it was heard.
def exercise_swaras(raaga: Raaga, topic: str) -> Tuple[List[str], str]:
    """The notes for one topic, and the instrument to play them on."""
    if topic in ("identity", "varisai", "mood", "tempo"):
        return list(raaga.arohanam) + list(raaga.avarohanam), "flute"
    if topic == "prayoga":
        notes: List[str] = []
        for prayoga in raaga.prayogas[:3]:
            notes.extend(prayoga)
        return notes or list(raaga.arohanam), "veena"
    if topic == "gamaka":
        ornamented = [s for s in raaga.allowed if s in raaga.gamaka]
        base = ornamented or list(raaga.jeeva) or list(raaga.arohanam)
        return [s for s in base for _ in (0, 1)], "violin"
    if topic == "jeeva":
        seed = list(raaga.jeeva) + list(raaga.nyasa)
        return seed or list(raaga.arohanam), "veena"
    if topic == "alapana":
        notes = list(raaga.arohanam)
        for prayoga in raaga.prayogas[:2]:
            notes.extend(prayoga)
        notes.extend(raaga.avarohanam)
        return notes, "flute"
    if topic in ("avoid", "comparison", "structure"):
        notes = list(raaga.arohanam) + list(raaga.avarohanam)
        for prayoga in raaga.prayogas[:1]:
            notes.extend(prayoga)
        return notes, "veena"
    return list(raaga.arohanam) + list(raaga.avarohanam), "flute"


def render_exercise(raaga: Raaga, swaras: Sequence[str], instrument: str,
                    sample_rate: int = ANALYSIS_SR,
                    tonic_midi: int = DEFAULT_TONIC_MIDI
                    ) -> Tuple[np.ndarray, int]:
    inst = catalog.get(instrument) or catalog.get("flute")
    notes: List[Note] = []
    t = 0.0
    for swara in swaras:
        notes.append(Note(swara=swara, midi=raaga.midi(swara, tonic_midi),
                          start=t, duration=0.42, velocity=92))
        t += 0.52
    audio = render_notes(notes, inst, sample_rate, total_seconds=t + 0.4)
    return audio, sample_rate


# --------------------------------------------------------------------------
# transcripts
# --------------------------------------------------------------------------
_TIMECODE = re.compile(
    r"^\s*(?:\d+\s*$|\d{1,2}:\d{2}(?::\d{2})?[.,]?\d*\s*-->|WEBVTT)")


class TranscriptService:
    """Reads captions or notes the creator supplied. Fetches nothing."""

    @staticmethod
    def clean(text: str) -> str:
        """Strip caption scaffolding so the words are left on their own."""
        lines: List[str] = []
        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line or _TIMECODE.match(line):
                continue
            line = re.sub(r"</?[a-zA-Z][^>]*>", "", line)      # caption tags
            line = re.sub(r"\[[^\]]{0,40}\]", "", line)        # [music]
            line = line.strip()
            if line:
                lines.append(line)
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    @classmethod
    def read(cls, path: Path) -> str:
        return cls.clean(path.read_text(encoding="utf-8", errors="replace"))


# --------------------------------------------------------------------------
# segmentation (phase C)
# --------------------------------------------------------------------------
class ContentSegmenter:
    """Breaks long content into pieces that keep their place in the source."""

    @staticmethod
    def segment_text(text: str, words_per_segment: int = 90
                     ) -> List[Dict[str, Any]]:
        words = (text or "").split()
        if not words:
            return []
        segments: List[Dict[str, Any]] = []
        for start in range(0, len(words), words_per_segment):
            chunk = words[start:start + words_per_segment]
            segments.append({
                "kind": "text",
                "index": len(segments),
                "reference": f"words {start + 1}-{start + len(chunk)}",
                "text": " ".join(chunk),
            })
        return segments

    @staticmethod
    def segment_audio(result: analysis.AnalysisResult) -> List[Dict[str, Any]]:
        """One segment per heard phrase, keeping its timestamp - section 7C."""
        segments: List[Dict[str, Any]] = []
        for index, phrase in enumerate(result.phrases):
            swaras = [s for s in phrase.swaras if s]
            if len(swaras) < 2:
                continue
            segments.append({
                "kind": "phrase",
                "index": index,
                "reference": f"{phrase.start:.1f}s-{phrase.end:.1f}s",
                "start": round(phrase.start, 2),
                "end": round(phrase.end, 2),
                "swaras": swaras,
                "confidence": round(phrase.confidence, 3),
                "shape": phrase.shape(),
            })
        return segments


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------
class MediaIngestionService:
    """Phase B and C for whichever representation access permitted."""

    def __init__(self, raagas: RaagaLibrary, settings=None) -> None:
        self.raagas = raagas
        self.settings = settings

    def acquire(self, source: LearningSource, decision: AccessDecision,
                max_seconds: float = 120.0) -> AcquiredContent:
        if decision.representation == "exercise":
            return self._acquire_exercise(source)
        if decision.representation == "audio":
            return self._acquire_audio(source, max_seconds)
        if decision.representation == "transcript":
            return self._acquire_transcript(source)
        return AcquiredContent(
            representation="none",
            error=decision.reason or "no permitted representation")

    # ------------------------------------------------------------------
    def _acquire_exercise(self, source: LearningSource) -> AcquiredContent:
        name = str(source.metadata.get("raaga", ""))
        raaga = self.raagas.get(name)
        if raaga is None:
            return AcquiredContent(
                representation="none",
                error=f"the library has no raaga called '{name}'")
        topic = str(source.metadata.get("topic", "identity"))
        swaras, instrument = exercise_swaras(raaga, topic)
        try:
            audio, sr = render_exercise(raaga, swaras, instrument)
        except Exception as exc:  # noqa: BLE001
            return AcquiredContent(representation="none",
                                   error=f"could not render the exercise: {exc}")
        return AcquiredContent(
            representation="exercise", audio=audio, sample_rate=sr,
            raaga=raaga, tonic_midi=float(DEFAULT_TONIC_MIDI),
            expected_swaras=list(swaras), language="swara notation")

    # ------------------------------------------------------------------
    def _acquire_audio(self, source: LearningSource,
                       max_seconds: float) -> AcquiredContent:
        path = Path(source.local_path or source.url.replace("file://", ""))
        try:
            audio, sr = analysis.load_audio(path, ANALYSIS_SR)
        except Exception as exc:  # noqa: BLE001
            return AcquiredContent(representation="none",
                                   error=f"could not read {path.name}: {exc}")
        if max_seconds > 0 and len(audio) > int(max_seconds * sr):
            audio = audio[:int(max_seconds * sr)]

        content = AcquiredContent(representation="audio", audio=audio,
                                  sample_rate=sr)
        named = str(source.metadata.get("raaga", ""))
        if named:
            content.raaga = self.raagas.get(named)

        # A real recording: prepare it before the ears see it.
        try:
            prepared = preprocess.prepare(audio, sr)
        except Exception as exc:  # noqa: BLE001 - never lose a source to this
            log.warning("could not prepare %s: %s", path.name, exc)
            content.warnings.append(f"preparation failed ({exc}); the audio "
                                    f"was analysed as it arrived")
            return content
        content.prepared = prepared
        content.audio = prepared.audio
        content.warnings.extend(prepared.warnings)
        if prepared.tonic_midi is not None:
            content.tonic_midi = prepared.tonic_midi
        return content

    # ------------------------------------------------------------------
    def _acquire_transcript(self, source: LearningSource) -> AcquiredContent:
        text = str(source.metadata.get("transcript", "") or "")
        if not text and source.local_path:
            try:
                text = TranscriptService.read(Path(source.local_path))
            except Exception as exc:  # noqa: BLE001
                return AcquiredContent(
                    representation="none",
                    error=f"could not read the transcript: {exc}")
        text = TranscriptService.clean(text)
        if not text:
            return AcquiredContent(representation="none",
                                   error="the transcript was empty")
        content = AcquiredContent(representation="transcript", transcript=text,
                                  language=source.language)
        content.segments = ContentSegmenter.segment_text(text)
        named = str(source.metadata.get("raaga", ""))
        if named:
            content.raaga = self.raagas.get(named)
        return content
