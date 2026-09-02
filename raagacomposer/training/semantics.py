"""Phases D and E - hearing the music, and working out what it taught.

Specification section 7.  Phase D extracts only what the evidence supports:
the specification is explicit that not every category should be forced onto
every source, so a transcript that never mentions tala produces no tala
finding rather than an empty one.

Phase E is the part that makes a Learning Report worth reading.  It separates
what was *observed* from what is *claimed*, and it keeps two things apart that
are easy to blur:

    understood      the shape of the lesson - what it was about, what it
                    covered, how it went about it
    learned         specific statements that can be written down, checked
                    against what we already hold, and traced to evidence

An observation here is never a fact yet.  It carries its evidence and its
confidence to :mod:`validation`, which decides whether it may be stored, and
what to do when it disagrees with something we already believe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agent import analysis
from ..core.logging_setup import get_logger
from ..raaga.library import Raaga, RaagaLibrary, parse_swara
from .acquisition import AcquiredContent, ContentSegmenter
from .models import LearningSource, Objective

log = get_logger("training.semantics")

#: Swara tokens as a teacher writes them, and as they are spoken.
_SWARA_TOKEN = re.compile(r"\b([SRGMPDN][1-3]?)\b")
_SWARA_WORD = re.compile(
    r"\b(sa|ri|ru|ga|gi|ma|mi|pa|dha|da|ni|nu)\b", re.IGNORECASE)
_WORD_TO_SWARA = {"sa": "S", "ri": "R2", "ru": "R2", "ga": "G3", "gi": "G3",
                  "ma": "M1", "mi": "M1", "pa": "P", "dha": "D2", "da": "D2",
                  "ni": "N3", "nu": "N3"}

TALA_NAMES = ("adi", "rupaka", "misra chapu", "khanda chapu", "triputa",
              "jhampa", "ata", "eka", "dhruva", "matya")

GAMAKA_WORDS = ("gamaka", "kampita", "jaru", "sphurita", "nokku", "odukkal",
                "orikai", "ravai", "khandippu", "oscillation", "slide")


@dataclass
class Observation:
    """One thing the source appears to say, with what supports it."""

    statement: str = ""
    subject: str = ""
    concept: str = ""
    category: str = ""
    raga: str = ""
    tala: str = ""
    difficulty: str = ""
    evidence: str = ""
    confidence: float = 0.0
    reference: str = ""
    objective_category: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Interpretation:
    """Everything phase E concluded about one source."""

    observations: List[Observation] = field(default_factory=list)
    understood: str = ""
    summary: str = ""
    limits: List[str] = field(default_factory=list)
    analysis: Optional[analysis.AnalysisResult] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    heard_accuracy: float = 0.0


class SemanticLearningService:
    """Turns content into checkable statements - phases D and E."""

    def __init__(self, raagas: RaagaLibrary) -> None:
        self.raagas = raagas

    # ==================================================================
    def interpret(self, source: LearningSource, content: AcquiredContent,
                  objectives: Sequence[Objective]) -> Interpretation:
        if content.representation == "transcript":
            return self._from_transcript(source, content)
        if content.has_audio:
            return self._from_audio(source, content)
        return Interpretation(
            understood="Nothing could be analysed.",
            summary="No usable representation of this source was available.",
            limits=["the content was never analysed"])

    # ==================================================================
    # audio
    # ==================================================================
    def _from_audio(self, source: LearningSource,
                    content: AcquiredContent) -> Interpretation:
        raaga = content.raaga
        try:
            result = analysis.analyse(
                content.audio, content.sample_rate, raaga,
                fixed_tonic_midi=content.tonic_midi)
        except Exception as exc:  # noqa: BLE001
            return Interpretation(
                understood="The audio could not be analysed.",
                summary=f"analysis failed: {exc}",
                limits=[f"analysis failed: {exc}"])

        interpretation = Interpretation(analysis=result)
        interpretation.segments = ContentSegmenter.segment_audio(result)
        if not result.notes:
            interpretation.understood = (
                "Nothing pitched could be made out in this recording.")
            interpretation.summary = "; ".join(result.warnings) or \
                "no notes were heard"
            interpretation.limits.append("no notes were heard")
            return interpretation

        heard = [n.swara for n in result.notes if n.swara]
        bases = [parse_swara(s)[0] for s in heard]
        distinct = sorted(set(bases))
        name = raaga.name if raaga is not None else ""
        reference = f"0.0s-{result.duration:.1f}s"

        # -- what the ears actually got ---------------------------------
        from_drone = (content.prepared is not None
                      and content.prepared.drone.found)
        how = "taken from the drone" if from_drone else "estimated from the melody"
        interpretation.observations.append(Observation(
            statement=(f"The tonic of this recording sits at "
                       f"{result.tonic_hz:.1f} Hz (MIDI "
                       f"{result.tonic_midi:.1f})."),
            subject=name or source.title, concept="tonic", category="tonic",
            raga=name,
            evidence=f"{how}; {len(result.notes)} notes examined",
            confidence=round(result.confidence, 3), reference=reference,
            objective_category="raaga", tags=["measured"]))

        if result.tempo_bpm > 0:
            interpretation.observations.append(Observation(
                statement=f"The material moves at about "
                          f"{result.tempo_bpm:.0f} bpm.",
                subject=name or source.title, concept="tempo",
                category="tempo", raga=name,
                evidence=f"measured across {len(result.notes)} notes",
                confidence=round(min(0.8, result.confidence), 3),
                reference=reference, objective_category="tempo",
                tags=["measured"]))

        interpretation.observations.append(Observation(
            statement=f"The swaras heard are {' '.join(distinct)}.",
            subject=name or source.title, concept="swaras heard",
            category="scale", raga=name,
            evidence=f"{len(heard)} notes identified by ear",
            confidence=round(result.confidence, 3), reference=reference,
            objective_category="scale", tags=["measured"]))

        # -- phrases, which are the point of listening -------------------
        for segment in interpretation.segments[:8]:
            swaras = segment["swaras"]
            if len(swaras) < 3:
                continue
            interpretation.observations.append(Observation(
                statement=f"A phrase of the raaga: {' '.join(swaras)}.",
                subject=name or source.title, concept="phrase",
                category="phrase", raga=name,
                evidence=f"heard at {segment['reference']}, shape "
                         f"{segment['shape']}",
                confidence=round(float(segment["confidence"]), 3),
                reference=segment["reference"],
                objective_category="phrase", tags=["heard", segment["shape"]]))

        # -- did we hear what was played? --------------------------------
        if content.expected_swaras:
            interpretation.heard_accuracy = self._agreement(
                bases, [parse_swara(s)[0] for s in content.expected_swaras])
            interpretation.observations.append(Observation(
                statement=(f"Hearing this exercise back, "
                           f"{interpretation.heard_accuracy:.0%} of the swaras "
                           f"played were identified correctly."),
                subject=name or source.title, concept="ear accuracy",
                category="self-assessment", raga=name,
                evidence=f"{len(content.expected_swaras)} swaras played, "
                         f"{len(heard)} heard",
                confidence=round(result.confidence, 3), reference=reference,
                objective_category="raaga", tags=["self-assessment"]))

        # -- does what we heard match what the library asserts? ----------
        if raaga is not None:
            outside = sorted({b for b in bases if b not in set(raaga.allowed)})
            if outside:
                interpretation.observations.append(Observation(
                    statement=(f"{len(outside)} swara(s) outside "
                               f"{raaga.name} were heard: "
                               f"{', '.join(outside)}."),
                    subject=raaga.name, concept="swaras outside the raaga",
                    category="grammar", raga=raaga.name,
                    evidence="compared against the shipped raaga definition",
                    confidence=round(0.5 * result.confidence, 3),
                    reference=reference, objective_category="grammar",
                    tags=["disagreement"]))

        interpretation.understood = self._understood_audio(
            source, content, result, distinct, interpretation)
        interpretation.summary = (
            f"{content.describe()}. {len(result.notes)} notes and "
            f"{len(result.phrases)} phrases were made out, tonic "
            f"{result.tonic_hz:.0f} Hz, confidence "
            f"{result.confidence:.2f}.")
        if content.prepared is not None and content.prepared.silenced_seconds:
            interpretation.limits.append(
                f"{content.prepared.silenced_seconds:.0f}s of this recording "
                f"was talking rather than singing and was not analysed")
        if result.warnings:
            interpretation.limits.extend(result.warnings)
        return interpretation

    # ------------------------------------------------------------------
    @staticmethod
    def _agreement(heard: Sequence[str], played: Sequence[str]) -> float:
        """How much of what was played turned up in what was heard."""
        if not played:
            return 0.0
        wanted, got = set(played), set(heard)
        return round(len(wanted & got) / len(wanted), 3)

    def _understood_audio(self, source: LearningSource,
                          content: AcquiredContent,
                          result: analysis.AnalysisResult,
                          distinct: Sequence[str],
                          interpretation: Interpretation) -> str:
        """Concepts, not a transcript - section 8.4."""
        parts: List[str] = []
        name = content.raaga.name if content.raaga is not None else ""
        if content.representation == "exercise":
            parts.append(
                f"This was an exercise the system set itself: it played "
                f"{len(content.expected_swaras)} swaras from its own "
                f"definition of {name or 'the raaga'} and then listened back "
                f"to what it had played.")
            if interpretation.heard_accuracy:
                parts.append(
                    f"It recognised {interpretation.heard_accuracy:.0%} of "
                    f"them by ear, which is a statement about its hearing "
                    f"rather than about the raaga.")
        else:
            parts.append(
                f"This is a recording supplied by the creator, "
                f"{result.duration:.0f} seconds of it.")
            if content.prepared is not None:
                drone = content.prepared.drone
                if drone.found:
                    parts.append(
                        f"A drone runs underneath at {drone.hz:.0f} Hz, and "
                        f"the tonic was taken from it rather than guessed at.")
                if content.prepared.silenced_seconds:
                    parts.append(
                        f"About {content.prepared.silenced_seconds:.0f} "
                        f"seconds of it is speech, which was set aside so "
                        f"only the singing was analysed.")
        parts.append(
            f"The material moves within {', '.join(distinct)} and falls into "
            f"{len(result.phrases)} phrases.")
        if name:
            disagrees = any("disagreement" in o.tags
                            for o in interpretation.observations)
            verdict = ("do not entirely agree with" if disagrees
                       else "are consistent with")
            parts.append(
                f"Taken as {name}, the swaras heard {verdict} the definition "
                f"the application already holds.")
        return " ".join(parts)

    # ==================================================================
    # transcript
    # ==================================================================
    def _from_transcript(self, source: LearningSource,
                         content: AcquiredContent) -> Interpretation:
        text = content.transcript
        lower = text.lower()
        interpretation = Interpretation(segments=content.segments)

        raaga = content.raaga
        if raaga is None:
            from .search import match_raaga
            raaga = match_raaga(f"{source.title} {text[:2000]}", self.raagas)
        name = raaga.name if raaga is not None else ""

        if name:
            interpretation.observations.append(Observation(
                statement=f"This lesson is about {name}.",
                subject=name, concept="raaga taught", category="raaga",
                raga=name, evidence="named in the title or the transcript",
                confidence=0.7, reference="whole source",
                objective_category="raaga", tags=["stated"]))

        # -- swara sequences the teacher spells out ----------------------
        for segment in content.segments:
            sequence = self._swara_sequence(segment["text"])
            if len(sequence) >= 4:
                interpretation.observations.append(Observation(
                    statement=f"A sequence given in the lesson: "
                              f"{' '.join(sequence)}.",
                    subject=name or source.title, concept="swara sequence",
                    category="phrase", raga=name,
                    evidence=f"spelled out at {segment['reference']}",
                    confidence=0.55, reference=segment["reference"],
                    objective_category="phrase", tags=["stated"]))

        # -- tala, only if it is actually mentioned ----------------------
        for tala in TALA_NAMES:
            if re.search(rf"\b{re.escape(tala)}\b\s*(tala|talam)?", lower):
                interpretation.observations.append(Observation(
                    statement=f"The lesson refers to {tala.title()} tala.",
                    subject=name or source.title, concept="tala",
                    category="tala", raga=name, tala=tala.title(),
                    evidence=f"'{tala}' appears in the transcript",
                    confidence=0.5, reference="whole source",
                    objective_category="tala", tags=["stated"]))
                break

        # -- gamaka vocabulary -------------------------------------------
        mentioned = sorted({w for w in GAMAKA_WORDS if w in lower})
        if mentioned:
            interpretation.observations.append(Observation(
                statement=f"The lesson discusses ornamentation: "
                          f"{', '.join(mentioned)}.",
                subject=name or source.title, concept="gamaka",
                category="ornament", raga=name,
                evidence=f"terms used: {', '.join(mentioned)}",
                confidence=0.45, reference="whole source",
                objective_category="ornament", tags=["stated"]))

        words = len(text.split())
        interpretation.understood = (
            f"This is a spoken or written lesson of about {words} words"
            + (f", about {name}" if name else "")
            + ". What it says has been read; nothing was heard, so any claim "
              "here rests on what the teacher stated rather than on anything "
              "the system listened to.")
        interpretation.summary = (
            f"Transcript of {words} words in "
            f"{len(content.segments)} segment(s); "
            f"{len(interpretation.observations)} statement(s) extracted.")
        interpretation.limits.append(
            "read as text only - no audio from this source was analysed, so "
            "nothing here has been verified by ear")
        return interpretation

    # ------------------------------------------------------------------
    @staticmethod
    def _swara_sequence(text: str) -> List[str]:
        """Swaras a teacher spelled out, in either notation."""
        tokens = _SWARA_TOKEN.findall(text)
        if len(tokens) >= 4:
            return tokens[:16]
        spoken = [_WORD_TO_SWARA[w.lower()]
                  for w in _SWARA_WORD.findall(text)
                  if w.lower() in _WORD_TO_SWARA]
        return spoken[:16] if len(spoken) >= 4 else []
