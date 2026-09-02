"""Research, ingestion and knowledge extraction (learning specification 6, 7, 18).

Sources come from providers, each of which declares what rights it has to the
material it offers.  Nothing here bypasses DRM, paywalls or access controls,
and nothing is downloaded from a platform that has not offered it: the web
provider records *leads* with their provenance and asks the creator to supply
material they are entitled to, rather than helping itself.

The providers that work on day one need no network at all:

``reference``   material the agent renders for itself from the structural raaga
                library, the way a student plays a scale to hear it;
``corpus``      audio files the creator puts in their own learning folder;
``project``     the application's own renders of its own compositions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging_setup import get_logger
from ..core.models import Note
from ..core.settings import Settings
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..raaga.library import (SWARA_SEMITONES, Raaga, RaagaLibrary, parse_swara)
from . import analysis, preprocess
from .knowledge import Fact, KnowledgeRepository, Phrase, Source

log = get_logger("agent.research")

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
ANALYSIS_SR = analysis.DEFAULT_SR


@dataclass
class SourceCandidate:
    locator: str
    title: str
    provider: str
    raaga: str = ""
    performer: str = ""
    content_type: str = "audio"
    rights_status: str = "unknown"
    quality: float = 0.5
    notes: str = ""
    tonic_midi: Optional[float] = None
    audio_loader: Optional[Callable[[], Tuple[np.ndarray, int]]] = None
    ingestable: bool = True

    def describe(self) -> str:
        return f"[{self.provider}] {self.title} ({self.rights_status})"


@dataclass
class IngestionResult:
    source_id: str = ""
    candidate: Optional[SourceCandidate] = None
    analysed: bool = False
    phrases_learned: int = 0
    phrases_rejected: int = 0
    facts_learned: int = 0
    confidence: float = 0.0
    error: str = ""
    result: Optional[analysis.AnalysisResult] = None
    prepared: Optional[preprocess.PreparedAudio] = None

    def summary(self) -> str:
        if self.error:
            return f"failed: {self.error}"
        text = (f"{self.phrases_learned} phrase(s) learned, "
                f"{self.phrases_rejected} rejected, "
                f"{self.facts_learned} fact(s), confidence {self.confidence:.2f}")
        if self.prepared is not None:
            text += f" [{self.prepared.summary()}]"
        return text


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
class SourceProvider:
    name = "provider"
    rights_status = "unknown"

    def find(self, raaga: Raaga, goal: str, limit: int) -> List[SourceCandidate]:
        return []


class ReferenceProvider(SourceProvider):
    """Material the agent renders for itself from the structural library.

    This is practice, not borrowing: the notes come from the raaga definition
    the application already ships, and the agent listens to its own playing to
    learn to hear.  Rights are unambiguous and it works offline.
    """

    name = "reference"
    rights_status = "internally-generated"

    def __init__(self, sample_rate: int = ANALYSIS_SR, tonic_midi: int = 60):
        self.sample_rate = sample_rate
        self.tonic_midi = tonic_midi

    def find(self, raaga: Raaga, goal: str, limit: int) -> List[SourceCandidate]:
        exercises: List[Tuple[str, List[str], str, float]] = [
            ("arohanam", list(raaga.arohanam), "flute", 0.95),
            ("avarohanam", list(raaga.avarohanam), "flute", 0.95),
        ]
        for i, prayoga in enumerate(raaga.prayogas):
            exercises.append((f"prayoga {i + 1}", list(prayoga), "veena", 0.85))
        # Joined phrases teach how the raaga moves between its own idioms.
        for i in range(len(raaga.prayogas) - 1):
            joined = list(raaga.prayogas[i]) + list(raaga.prayogas[i + 1])
            exercises.append((f"prayoga pair {i + 1}", joined, "violin", 0.75))

        # Offer everything it can play; the agent decides how much to take in
        # one lesson, and comes back for the rest next time.
        candidates = []
        for label, swaras, instrument, quality in exercises:
            candidates.append(SourceCandidate(
                locator=f"reference://{raaga.name}/{label.replace(' ', '-')}",
                title=f"{raaga.name} {label}",
                provider=self.name,
                raaga=raaga.name,
                performer="the agent itself",
                content_type="rendered-exercise",
                rights_status=self.rights_status,
                quality=quality,
                tonic_midi=float(self.tonic_midi),
                notes=" ".join(swaras),
                audio_loader=self._make_loader(raaga, swaras, instrument)))
        return candidates

    def _make_loader(self, raaga: Raaga, swaras: Sequence[str],
                     instrument_key: str) -> Callable[[], Tuple[np.ndarray, int]]:
        def load() -> Tuple[np.ndarray, int]:
            inst = catalog.get(instrument_key) or catalog.get("flute")
            notes: List[Note] = []
            t = 0.0
            for swara in swaras:
                notes.append(Note(swara=swara,
                                  midi=raaga.midi(swara, self.tonic_midi),
                                  start=t, duration=0.42, velocity=92))
                t += 0.52
            audio = render_notes(notes, inst, self.sample_rate,
                                 total_seconds=t + 0.4)
            return audio, self.sample_rate
        return load


class LocalCorpusProvider(SourceProvider):
    """Audio the creator placed in their own learning folder."""

    name = "corpus"
    rights_status = "user-supplied"

    def __init__(self, folder: Optional[Path]) -> None:
        self.folder = Path(folder) if folder else None

    def find(self, raaga: Raaga, goal: str, limit: int) -> List[SourceCandidate]:
        if not self.folder or not self.folder.exists():
            return []
        wanted = raaga.name.lower()
        candidates: List[SourceCandidate] = []
        for path in sorted(self.folder.rglob("*")):
            if path.suffix.lower() not in AUDIO_SUFFIXES or not path.is_file():
                continue
            blob = f"{path.parent.name} {path.stem}".lower()
            # A file is taken as this raaga if its name or folder says so.
            if wanted not in blob and not any(
                    alias.lower() in blob for alias in raaga.aliases):
                continue
            candidates.append(SourceCandidate(
                locator=str(path), title=path.stem, provider=self.name,
                raaga=raaga.name, content_type="audio",
                rights_status=self.rights_status, quality=0.8,
                notes=f"from {self.folder}",
                audio_loader=self._loader(path)))
            if len(candidates) >= limit:
                break
        return candidates

    @staticmethod
    def _loader(path: Path) -> Callable[[], Tuple[np.ndarray, int]]:
        def load() -> Tuple[np.ndarray, int]:
            return analysis.load_audio(path, ANALYSIS_SR)
        return load


class ProjectRenderProvider(SourceProvider):
    """The application's own renders - it can listen back to what it made."""

    name = "project"
    rights_status = "own-output"

    def __init__(self, projects_dir: Optional[Path]) -> None:
        self.projects_dir = Path(projects_dir) if projects_dir else None

    def find(self, raaga: Raaga, goal: str, limit: int) -> List[SourceCandidate]:
        if not self.projects_dir or not self.projects_dir.exists():
            return []
        candidates: List[SourceCandidate] = []
        for path in sorted(self.projects_dir.glob("*/audio/tune_v*.wav")):
            candidates.append(SourceCandidate(
                locator=str(path), title=f"own tune: {path.parent.parent.name}",
                provider=self.name, raaga=raaga.name, content_type="audio",
                rights_status=self.rights_status, quality=0.6,
                notes="the agent's own earlier composition",
                audio_loader=LocalCorpusProvider._loader(path)))
            if len(candidates) >= limit:
                break
        return candidates


class WebLeadProvider(SourceProvider):
    """Records leads to public material; it does not fetch anything.

    Public availability is not permission.  This provider stores where useful
    material is said to exist, with its rights status marked unverified, so the
    creator can supply anything they are entitled to use.  Disabled unless the
    creator turns it on.
    """

    name = "web"
    rights_status = "external-unverified"

    def __init__(self, enabled: bool = False, llm=None) -> None:
        self.enabled = enabled
        self.llm = llm

    def find(self, raaga: Raaga, goal: str, limit: int) -> List[SourceCandidate]:
        if not self.enabled:
            return []
        leads: List[SourceCandidate] = []
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                text = self.llm.explain(
                    f"Name up to {limit} well-known, authoritative recorded "
                    f"examples or teaching resources for the Carnatic raaga "
                    f"{raaga.name} ({goal}). One per line as "
                    f"'performer - title'. No links.",
                    context="")
                for line in [l.strip("-* ") for l in text.splitlines() if l.strip()]:
                    performer, _, title = line.partition(" - ")
                    leads.append(SourceCandidate(
                        locator=f"lead://{raaga.name}/{line[:60]}",
                        title=title or line, performer=performer,
                        provider=self.name, raaga=raaga.name,
                        content_type="lead", rights_status=self.rights_status,
                        quality=0.4, ingestable=False,
                        notes="lead only - supply the audio yourself if you are "
                              "entitled to use it"))
                    if len(leads) >= limit:
                        break
            except Exception as exc:  # noqa: BLE001
                log.warning("web leads unavailable: %s", exc)
        return leads


# --------------------------------------------------------------------------
# research agent
# --------------------------------------------------------------------------
class ResearchAgent:
    def __init__(self, repository: KnowledgeRepository, library: RaagaLibrary,
                 settings: Optional[Settings] = None, llm=None) -> None:
        self.repo = repository
        self.library = library
        self.settings = settings or Settings.load()
        corpus = getattr(self.settings, "learning_corpus_dir", "")
        self.providers: List[SourceProvider] = [
            LocalCorpusProvider(Path(corpus) if corpus else None),
            ReferenceProvider(),
            ProjectRenderProvider(Path(self.settings.projects_dir)
                                  if self.settings.projects_dir else None),
            WebLeadProvider(bool(getattr(self.settings, "learning_allow_web", False)),
                            llm),
        ]

    # -- discovery ---------------------------------------------------------
    def find_sources(self, raaga_name: str, goal: str = "phrases",
                     limit: int = 6, skip_known: bool = True
                     ) -> List[SourceCandidate]:
        raaga = self.library.get(raaga_name)
        if raaga is None:
            return []
        found: List[SourceCandidate] = []
        for provider in self.providers:
            try:
                candidates = provider.find(raaga, goal, limit)
            except Exception as exc:  # noqa: BLE001
                log.warning("provider %s failed: %s", provider.name, exc)
                continue
            for candidate in candidates:
                if skip_known and self.repo.has_source(candidate.provider,
                                                       candidate.locator):
                    continue
                found.append(candidate)
        # Best rights and best quality first: user-supplied audio outranks
        # self-rendered practice material, which outranks an unverified lead.
        rank = {"user-supplied": 0, "own-output": 1, "internally-generated": 2,
                "external-unverified": 9}
        found.sort(key=lambda c: (rank.get(c.rights_status, 5), -c.quality))
        return found[:limit]

    # -- ingestion ---------------------------------------------------------
    def _should_preprocess(self, candidate: SourceCandidate) -> bool:
        """Only audio a person supplied needs preparing.

        Everything else the agent listens to it rendered itself, so it is
        already one clean voice with no drone and nobody talking over it.
        Running the gate on that material could only take good phrases away.
        """
        if not getattr(self.settings, "learning_preprocess_recordings", True):
            return False
        return (candidate.content_type == "audio"
                and candidate.rights_status == "user-supplied")

    def ingest(self, candidate: SourceCandidate,
               max_seconds: Optional[float] = None) -> IngestionResult:
        result = IngestionResult(candidate=candidate)
        raaga = self.library.get(candidate.raaga)
        source = Source(
            locator=candidate.locator, title=candidate.title,
            performer=candidate.performer, raaga=candidate.raaga,
            content_type=candidate.content_type,
            rights_status=candidate.rights_status, provider=candidate.provider,
            quality=candidate.quality, extraction_version=analysis.ANALYSIS_VERSION,
            notes=candidate.notes)
        stored, is_new = self.repo.add_source(source)
        result.source_id = stored.id
        if not is_new:
            result.error = "already ingested"
            return result

        if not candidate.ingestable or candidate.audio_loader is None:
            self.repo.update_source(stored.id, status="lead_only",
                                    notes="recorded as a lead; no material fetched")
            result.error = "lead recorded; no audio was fetched"
            return result

        try:
            audio, sr = candidate.audio_loader()
        except Exception as exc:  # noqa: BLE001
            self.repo.update_source(stored.id, status="failed", error=str(exc))
            result.error = f"could not read the source: {exc}"
            log.warning("ingest failed for %s: %s", candidate.locator, exc)
            return result

        limit = max_seconds if max_seconds is not None else float(
            getattr(self.settings, "learning_max_audio_seconds", 120))
        if limit > 0 and len(audio) > int(limit * sr):
            audio = audio[:int(limit * sr)]

        # A supplied recording is a real one: a teacher talking over a drone,
        # occasionally singing.  Prepare it before the ears see it.  Rendered
        # exercises and the application's own output need none of this and are
        # left exactly as they were.
        prepared = None
        tonic_hint = candidate.tonic_midi
        fixed_tonic = None
        if self._should_preprocess(candidate):
            try:
                prepared = preprocess.prepare(
                    audio, sr,
                    remove_drone=bool(getattr(self.settings,
                                              "learning_remove_drone", True)),
                    gate_speech=bool(getattr(self.settings,
                                             "learning_gate_speech", True)))
            except Exception as exc:  # noqa: BLE001 - never lose a source to this
                log.warning("preparation failed for %s, using the audio as it "
                            "arrived: %s", candidate.locator, exc)
            else:
                audio = prepared.audio
                result.prepared = prepared
                # The tanpura is the one thing in the room that knows Sa.
                if candidate.tonic_midi is None and prepared.tonic_midi is not None:
                    fixed_tonic = prepared.tonic_midi
                log.info("prepared %s: %s", candidate.title, prepared.summary())

        try:
            analysed = analysis.analyse(audio, sr, raaga, tonic_hint,
                                        fixed_tonic_midi=fixed_tonic)
        except Exception as exc:  # noqa: BLE001
            self.repo.update_source(stored.id, status="failed", error=str(exc))
            result.error = f"analysis failed: {exc}"
            return result

        if prepared is not None:
            self.repo.update_source(
                stored.id,
                extraction_version=f"{analysis.ANALYSIS_VERSION}+"
                                   f"{prepared.version}")

        result.result = analysed
        result.analysed = True
        result.confidence = analysed.confidence
        if analysed.warnings:
            log.info("analysis warnings for %s: %s", candidate.title,
                     "; ".join(analysed.warnings))
        if not analysed.notes:
            self.repo.update_source(stored.id, status="empty",
                                    error="; ".join(analysed.warnings) or
                                    "no notes were heard")
            result.error = "nothing musical was heard"
            return result

        learned, rejected, facts = self._extract(stored, candidate, analysed, raaga)
        result.phrases_learned = learned
        result.phrases_rejected = rejected
        result.facts_learned = facts
        self.repo.update_source(stored.id, status="analysed",
                                confidence=analysed.confidence)
        self.repo.log_event(
            "source.analysed",
            f"{candidate.title}: {result.summary()}",
            raaga=candidate.raaga, source_id=stored.id)
        return result

    # -- knowledge extraction ---------------------------------------------
    def _extract(self, source: Source, candidate: SourceCandidate,
                 analysed: analysis.AnalysisResult,
                 raaga: Optional[Raaga]) -> Tuple[int, int, int]:
        min_confidence = float(getattr(self.settings,
                                       "learning_min_confidence", 0.35))
        allowed = set(raaga.allowed) if raaga else set()
        learned = rejected = 0

        for phrase in analysed.phrases:
            # A held note arrives from the tracker as a run of identical swaras.
            # That is one note, not a melodic pattern, so collapse it before
            # anything is remembered as a phrase.
            swaras, durations = _collapse_repeats(phrase.swaras, phrase.durations)
            if len(swaras) < 3 or len(set(swaras)) < 3:
                rejected += 1
                continue
            confidence = round(phrase.confidence * 0.5
                               + analysed.confidence * 0.3
                               + source.quality * 0.2, 3)
            bases = [parse_swara(s)[0] for s in swaras]
            if allowed and any(b not in allowed for b in bases):
                rejected += 1
                continue
            if confidence < min_confidence:
                rejected += 1
                continue
            record = Phrase(
                raaga=source.raaga, swaras=list(swaras), midi=phrase.midi,
                durations=durations, function=self._function(phrase),
                source_id=source.id, confidence=confidence,
                contour=phrase.contour(), tempo=analysed.tempo_bpm,
                notes=candidate.title[:120])
            _, is_new = self.repo.add_phrase(record)
            learned += int(is_new)

        facts = 0
        if raaga is not None and analysed.notes:
            facts += self._observe_facts(source, analysed, raaga)
        return learned, rejected, facts

    @staticmethod
    def _function(phrase: analysis.AnalysedPhrase) -> str:
        shape = phrase.shape()
        return {"rise": "ascent", "fall": "descent"}.get(shape, shape or "phrase")

    def _observe_facts(self, source: Source, analysed: analysis.AnalysisResult,
                       raaga: Raaga) -> int:
        """Write down what was actually heard, with its provenance."""
        facts = 0
        heard = [parse_swara(n.swara)[0] for n in analysed.notes]
        if not heard:
            return 0

        ascending = [b for a, b in zip(analysed.notes, analysed.notes[1:])
                     if b.midi > a.midi]
        descending = [b for a, b in zip(analysed.notes, analysed.notes[1:])
                      if b.midi < a.midi]

        def order(notes) -> List[str]:
            seen: List[str] = []
            for note in notes:
                base = parse_swara(note.swara)[0]
                if base not in seen:
                    seen.append(base)
            return sorted(seen, key=lambda s: SWARA_SEMITONES.get(s, 0))

        confidence = round(min(0.9, analysed.confidence * source.quality + 0.1), 3)
        if len(ascending) >= 3:
            self.repo.add_fact(Fact(
                raaga=source.raaga, key="observed_ascent",
                value=" ".join(order(ascending)), confidence=confidence,
                source_id=source.id,
                notes=f"heard in {source.title}"))
            facts += 1
        if len(descending) >= 3:
            self.repo.add_fact(Fact(
                raaga=source.raaga, key="observed_descent",
                value=" ".join(order(descending)), confidence=confidence,
                source_id=source.id, notes=f"heard in {source.title}"))
            facts += 1

        endings = [parse_swara(p.swaras[-1])[0] for p in analysed.phrases
                   if p.swaras]
        if endings:
            counts: Dict[str, int] = {}
            for swara in endings:
                counts[swara] = counts.get(swara, 0) + 1
            resting = sorted(counts, key=counts.get, reverse=True)[:3]
            self.repo.add_fact(Fact(
                raaga=source.raaga, key="observed_resting_notes",
                value=" ".join(resting), confidence=confidence,
                source_id=source.id,
                notes=f"phrase endings in {source.title}"))
            facts += 1
        if analysed.tempo_bpm:
            self.repo.add_fact(Fact(
                raaga=source.raaga, key="observed_tempo",
                value=f"{analysed.tempo_bpm:.0f}", confidence=confidence * 0.7,
                source_id=source.id, notes=source.title))
            facts += 1
        return facts

    # -- structural seeding ------------------------------------------------
    def seed_structural_knowledge(self, raaga_name: str) -> int:
        """Write the shipped structural facts into memory, with provenance.

        The library that ships with the application is itself a source: a
        reference book the student was given.  Recording it as such means the
        agent can say where a fact came from and how sure it is.
        """
        raaga = self.library.get(raaga_name)
        if raaga is None:
            return 0
        source = Source(
            locator=f"library://{raaga.name}", title=f"{raaga.name} (reference book)",
            performer="shipped raaga library", raaga=raaga.name,
            content_type="structural", rights_status="internally-generated",
            provider="library", quality=0.9, confidence=0.9, status="analysed",
            extraction_version=analysis.ANALYSIS_VERSION,
            notes="the structural definition the application ships with")
        stored, _ = self.repo.add_source(source)

        pairs = [
            ("name", raaga.name),
            ("aliases", ", ".join(raaga.aliases) or "none"),
            ("arohanam", " ".join(raaga.arohanam)),
            ("avarohanam", " ".join(raaga.avarohanam)),
            ("swaras", " ".join(raaga.allowed)),
            ("jeeva", " ".join(raaga.jeeva)),
            ("nyasa", " ".join(raaga.nyasa)),
            ("graha", " ".join(raaga.graha)),
            ("gamaka", ", ".join(f"{k}:{v}" for k, v in raaga.gamaka.items())),
            ("moods", ", ".join(raaga.moods)),
            ("tempo_range", f"{raaga.tempo_range[0]}-{raaga.tempo_range[-1]}"),
            ("melakarta", str(raaga.melakarta or "janya")),
            ("notes", raaga.notes),
        ]
        written = 0
        for key, value in pairs:
            if not value:
                continue
            self.repo.add_fact(Fact(raaga=raaga.name, key=key, value=value,
                                    confidence=0.9, source_id=stored.id,
                                    notes="from the shipped library"))
            written += 1
        self.repo.log_event("knowledge.seeded",
                            f"{written} structural facts for {raaga.name}",
                            raaga=raaga.name, source_id=stored.id)
        return written


def _collapse_repeats(swaras, durations):
    """Merge consecutive identical swaras into one note with the total length."""
    out_swaras = []
    out_durations = []
    for i, swara in enumerate(swaras):
        length = durations[i] if i < len(durations) else 0.0
        if out_swaras and out_swaras[-1] == swara:
            out_durations[-1] = round(out_durations[-1] + length, 3)
            continue
        out_swaras.append(swara)
        out_durations.append(round(length, 3))
    return out_swaras, out_durations
