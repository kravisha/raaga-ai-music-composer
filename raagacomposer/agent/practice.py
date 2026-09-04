"""The practice engine (learning specification section 13).

The agent does not only read about music: it plays something, listens to what
came out with its own ears, and is marked on it.  Every exercise here closes
that loop - synthesise, analyse, compare, score - so a curriculum unit can only
pass if the agent's hearing and its knowledge actually work.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging_setup import get_logger
from ..core.models import Note, Section, SectionKind
from ..core.settings import Settings
from ..music import instruments as catalog
from ..music.melody import MelodyOptions, clamp_token, generate_section_notes
from ..music.synth import render_notes
from ..music.theory import midi_to_freq
from ..raaga.library import SWARA_SEMITONES, Raaga, RaagaLibrary, parse_swara
from . import analysis
from .curriculum import Unit
from .evaluator import Evaluation, Evaluator, Finding
from .guidance import Guidance
from .knowledge import KnowledgeRepository
from .learned import learned_phrase_bank, learned_raaga
from .originality import PhraseIndex, check as check_originality

log = get_logger("agent.practice")

PRACTICE_SR = analysis.DEFAULT_SR
TONIC = 60

VARIANT_NAMES = {1: "R1", 2: "R2", 3: "R3", 4: "G3", 5: "M1", 6: "M2",
                 8: "D1", 9: "D2", 10: "N2", 11: "N3"}


def practice_seed(unit_id: str, attempt: int = 0, salt: str = "") -> int:
    """The seed for one attempt at one unit.

    Derived from the unit id and the attempt number, never from the clock or
    from ``hash()``: the same attempt at the same lesson sets the same
    exercises in every process (str hashes are salted per process), and each
    retry sets different ones.  A retry that replays the exercises the agent
    just failed is not a retry - with a clock-based seed every attempt made
    within the same second was identical, so a lesson that failed once failed
    the same way until the curriculum gave up on it.

    ``salt`` mixes in a third component (the agent factory's trainer uses
    "test" / "validation" / "hidden" so a test never reuses a practice seed);
    an empty salt (the default) reproduces the exact seed string used before
    this parameter existed, so REG-100's pinned values are unchanged.
    """
    text = (f"{unit_id}#{int(attempt)}#{salt}" if salt
            else f"{unit_id}#{int(attempt)}")
    seed = zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF
    return seed or 1


def drift_neighbours(library: RaagaLibrary, raaga: Raaga,
                     limit: int = 4) -> List[Raaga]:
    """The raagas this one is most likely to be confused with.

    Most swaras in common first.  On a tie, a raaga somebody curated comes
    ahead of a bare parent scale, and then the name, so the answer does not
    depend on what order the library happened to load.

    That tie-break earns its place since the Stage 1 pack put all 72
    melakartas in the library: dozens of raagas now tie on overlap, and
    ordered only by overlap the confusable neighbours a student actually
    needs - Shankarabharanam and Kambhoji against Kalyani - were pushed out
    by whichever scales sorted first.  A melakarta that genuinely shares more
    swaras is still the nearer neighbour and still wins; this only settles
    the ties.
    """
    allowed = set(raaga.allowed)
    others = [r for r in library.all() if r.name != raaga.name]
    return sorted(others,
                  key=lambda r: (-len(set(r.allowed) & allowed),
                                 r.scale_only, r.name))[:limit]


@dataclass
class ExerciseResult:
    name: str
    score: float
    passed: bool
    expected: str = ""
    heard: str = ""
    detail: str = ""


@dataclass
class PracticeReport:
    unit_id: str = ""
    skill_type: str = ""
    score: float = 0.0
    passed: bool = False
    exercises: List[ExerciseResult] = field(default_factory=list)
    detail: str = ""
    artifacts: List[List[Note]] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    # Every exercise's findings, not only the last one's: an attempt with
    # five exercises is marked on all five.
    findings: List[Finding] = field(default_factory=list)
    # What this attempt was told before it ran, if anything.
    guidance: Optional[Guidance] = None

    def summary(self) -> str:
        got = sum(1 for e in self.exercises if e.passed)
        return (f"{self.unit_id}: {'passed' if self.passed else 'not yet'} "
                f"({self.score:.2f}, {got}/{len(self.exercises)} exercises)")


class PracticeEngine:
    def __init__(self, repository: KnowledgeRepository, library: RaagaLibrary,
                 settings: Optional[Settings] = None) -> None:
        self.repo = repository
        self.library = library
        self.settings = settings or Settings.load()
        # Set for the duration of a run(), read by the handlers that
        # generate; empty (never None) so a handler need not guard it.
        self._guidance: Guidance = Guidance()

    # ==================================================================
    def run(self, unit: Unit, raaga_name: str = "", seed: int = 0,
            guidance: Optional[Guidance] = None) -> PracticeReport:
        rng = random.Random(seed or practice_seed(unit.id))
        name = unit.raaga_name or raaga_name
        raaga = None
        if name:
            raaga, _ = learned_raaga(self.repo, self.library, name)
            if raaga is None:
                raaga = self.library.get(name)

        handler = {
            "listen.compare": self._listen_compare,
            "listen.identify": self._listen_identify,
            "listen.transcribe": self._listen_transcribe,
            "generate.pattern": self._generate_pattern,
            "generate.section": self._generate_section,
            "recall.fact": self._recall_fact,
            "recall.phrases": self._recall_phrases,
            "classify.valid": self._classify_valid,
            "correct.phrase": self._correct_phrase,
        }.get(unit.skill_type)

        report = PracticeReport(unit_id=unit.id, skill_type=unit.skill_type,
                                guidance=guidance)
        if handler is None:
            report.detail = f"no practice handler for {unit.skill_type}"
            return report
        self._guidance = guidance or Guidance()
        try:
            handler(unit, raaga, rng, report)
        except Exception as exc:  # noqa: BLE001 - a failed lesson is data
            log.exception("practice failed for %s", unit.id)
            report.detail = f"practice raised {type(exc).__name__}: {exc}"
            report.score = 0.0
            report.passed = False
            return report
        finally:
            self._guidance = Guidance()

        if report.exercises:
            report.score = round(
                sum(e.score for e in report.exercises) / len(report.exercises), 4)
        report.passed = report.score >= unit.minimum_pass_score
        if not report.detail:
            report.detail = report.summary()
        return report

    # ==================================================================
    # hearing helpers
    # ==================================================================
    def _render(self, swaras: Sequence[str], raaga: Optional[Raaga],
                tonic: int = TONIC, instrument: str = "flute",
                duration: float = 0.4, gap: float = 0.12) -> np.ndarray:
        inst = catalog.get(instrument) or catalog.get("flute")
        notes: List[Note] = []
        t = 0.0
        for swara in swaras:
            midi = (raaga.midi(swara, tonic) if raaga is not None
                    else tonic + SWARA_SEMITONES.get(parse_swara(swara)[0], 0))
            notes.append(Note(swara=swara, midi=midi, start=t, duration=duration,
                              velocity=94))
            t += duration + gap
        return render_notes(notes, inst, PRACTICE_SR, total_seconds=t + 0.3)

    def _render_notes(self, notes: Sequence[Note], instrument: str = "veena"
                      ) -> np.ndarray:
        inst = catalog.get(instrument) or catalog.get("flute")
        total = max((n.start + n.duration for n in notes), default=1.0) + 0.4
        return render_notes(notes, inst, PRACTICE_SR, total_seconds=total)

    def _hear(self, audio: np.ndarray, raaga: Optional[Raaga] = None,
              tonic_hint: Optional[float] = float(TONIC),
              fixed_tonic: Optional[float] = None) -> analysis.AnalysisResult:
        return analysis.analyse(audio, PRACTICE_SR, raaga, tonic_hint, fixed_tonic)

    # ==================================================================
    # Stage A: listening
    # ==================================================================
    def _listen_compare(self, unit: Unit, raaga: Optional[Raaga],
                        rng: random.Random, report: PracticeReport) -> None:
        mode = unit.params.get("compare", "higher_lower")
        for i in range(unit.exercises):
            if mode == "tone_or_noise":
                report.exercises.append(self._exercise_tone_or_noise(rng, i))
            elif mode == "direction":
                report.exercises.append(self._exercise_direction(rng, raaga, i))
            else:
                report.exercises.append(self._exercise_higher_lower(
                    rng, int(unit.params.get("min_semitones", 2)), i))

    def _exercise_tone_or_noise(self, rng: random.Random, i: int) -> ExerciseResult:
        is_tone = rng.random() < 0.5
        if is_tone:
            audio = self._render(["S"], None, TONIC + rng.randint(-5, 7),
                                 duration=0.7)
        else:
            noise = np.random.default_rng(rng.randint(0, 10 ** 6)).standard_normal(
                int(PRACTICE_SR * 0.8)).astype(np.float32) * 0.3
            audio = noise
        heard = self._hear(audio, None, None)
        # A tone gives a stable pitch with real confidence; noise does not.
        answered_tone = heard.voiced_ratio > 0.4 and heard.confidence > 0.35
        correct = answered_tone == is_tone
        return ExerciseResult(
            name=f"tone-or-noise {i + 1}", score=1.0 if correct else 0.0,
            passed=correct, expected="tone" if is_tone else "noise",
            heard="tone" if answered_tone else "noise",
            detail=f"voiced {heard.voiced_ratio:.2f} conf {heard.confidence:.2f}")

    def _exercise_higher_lower(self, rng: random.Random, min_semitones: int,
                               i: int) -> ExerciseResult:
        first = TONIC + rng.randint(-6, 6)
        delta = rng.choice([-1, 1]) * rng.randint(min_semitones, 9)
        second = first + delta
        heard_first = self._hear(self._render_notes(
            [Note(swara="S", midi=first, start=0.0, duration=0.6)]), None, None)
        heard_second = self._hear(self._render_notes(
            [Note(swara="S", midi=second, start=0.0, duration=0.6)]), None, None)
        if not heard_first.notes or not heard_second.notes:
            return ExerciseResult(name=f"higher-lower {i + 1}", score=0.0,
                                  passed=False, detail="heard nothing")
        answer = "second" if heard_second.notes[0].midi > heard_first.notes[0].midi \
            else "first"
        truth = "second" if delta > 0 else "first"
        correct = answer == truth
        return ExerciseResult(
            name=f"higher-lower {i + 1}", score=1.0 if correct else 0.0,
            passed=correct, expected=truth, heard=answer,
            detail=f"{delta:+d} semitones")

    def _exercise_direction(self, rng: random.Random, raaga: Optional[Raaga],
                            i: int) -> ExerciseResult:
        raaga = raaga or self.library.require("Shankarabharanam")
        ascending = rng.random() < 0.5
        # The raw arohanam and avarohanam keep their octave marks, so the line
        # genuinely rises or falls instead of returning to the same Sa.
        ladder = list(raaga.arohanam if ascending else raaga.avarohanam)
        start = rng.randint(0, max(0, len(ladder) - 5))
        swaras = ladder[start:start + 5]
        audio = self._render(swaras, raaga)
        heard = self._hear(audio, raaga, fixed_tonic=float(TONIC))
        if len(heard.notes) < 2:
            return ExerciseResult(name=f"direction {i + 1}", score=0.0,
                                  passed=False, detail="heard too little")
        rise = heard.notes[-1].midi - heard.notes[0].midi
        answer = "up" if rise > 0 else "down"
        # Mark against what was actually played, not against the intent.
        played = raaga.midi(swaras[-1], TONIC) - raaga.midi(swaras[0], TONIC)
        truth = "up" if played > 0 else "down"
        correct = answer == truth
        return ExerciseResult(
            name=f"direction {i + 1}", score=1.0 if correct else 0.0,
            passed=correct, expected=truth, heard=answer,
            detail=" ".join(swaras))

    def _listen_identify(self, unit: Unit, raaga: Optional[Raaga],
                         rng: random.Random, report: PracticeReport) -> None:
        mode = unit.params.get("identify", "swara")
        raaga = raaga or self.library.require("Shankarabharanam")
        for i in range(unit.exercises):
            if mode == "tonic":
                report.exercises.append(self._exercise_tonic(
                    rng, raaga, float(unit.params.get("tolerance_semitones", 1.0)), i))
            elif mode == "swara":
                report.exercises.append(self._exercise_swara(rng, raaga, i))
            elif mode == "variant":
                report.exercises.append(self._exercise_variant(rng, i))
            elif mode == "interval":
                report.exercises.append(self._exercise_interval(
                    rng, float(unit.params.get("tolerance_semitones", 1.0)), i))
            elif mode == "tempo":
                report.exercises.append(self._exercise_tempo(
                    rng, raaga, float(unit.params.get("tolerance_ratio", 0.18)), i))

    def _exercise_tonic(self, rng: random.Random, raaga: Raaga, tolerance: float,
                        i: int) -> ExerciseResult:
        tonic = 60 + rng.randint(-4, 5)
        swaras = ["S"] + [s for s in raaga.ascending[1:5]] + ["S"]
        audio = self._render(swaras, raaga, tonic)
        heard = self._hear(audio, raaga, None)
        error = abs(heard.tonic_midi - tonic) if heard.tonic_midi else 99.0
        correct = error <= tolerance
        return ExerciseResult(
            name=f"find Sa {i + 1}",
            score=round(max(0.0, 1.0 - error / max(0.5, tolerance * 3)), 3)
            if correct else 0.0,
            passed=correct, expected=f"MIDI {tonic}",
            heard=f"MIDI {heard.tonic_midi:.1f}",
            detail=f"error {error:.2f} semitones")

    def _exercise_swara(self, rng: random.Random, raaga: Raaga,
                        i: int) -> ExerciseResult:
        swara = rng.choice(raaga.allowed)
        audio = self._render([swara], raaga, TONIC, duration=0.7)
        heard = self._hear(audio, raaga, fixed_tonic=float(TONIC))
        answer = heard.notes[0].swara if heard.notes else ""
        correct = parse_swara(answer)[0] == swara if answer else False
        return ExerciseResult(
            name=f"name the swara {i + 1}", score=1.0 if correct else 0.0,
            passed=correct, expected=swara, heard=answer or "nothing")

    def _exercise_variant(self, rng: random.Random, i: int) -> ExerciseResult:
        semitone, name = rng.choice([(1, "R1"), (2, "R2"), (3, "R3"),
                                     (8, "D1"), (9, "D2"),
                                     (10, "N2"), (11, "N3")])
        audio = self._render_notes(
            [Note(swara=name, midi=TONIC + semitone, start=0.0, duration=0.7)])
        heard = self._hear(audio, None, fixed_tonic=float(TONIC))
        if not heard.notes:
            return ExerciseResult(name=f"variant {i + 1}", score=0.0, passed=False,
                                  expected=name, heard="nothing")
        measured = int(round(heard.notes[0].midi - TONIC))
        answer = VARIANT_NAMES.get(measured, f"{measured} semitones")
        correct = measured == semitone
        return ExerciseResult(
            name=f"variant {i + 1}", score=1.0 if correct else 0.0,
            passed=correct, expected=name, heard=answer)

    def _exercise_interval(self, rng: random.Random, tolerance: float,
                           i: int) -> ExerciseResult:
        base = TONIC + rng.randint(-4, 4)
        gap = rng.randint(1, 12)
        audio = self._render_notes([
            Note(swara="S", midi=base, start=0.0, duration=0.9),
            Note(swara="S", midi=base + gap, start=1.3, duration=0.9)], "flute")
        heard = self._hear(audio, None, None)
        if len(heard.notes) < 2:
            return ExerciseResult(name=f"interval {i + 1}", score=0.0, passed=False,
                                  expected=f"{gap}", heard="heard one note")
        steady = [n for n in heard.notes if n.duration >= 0.3] or heard.notes
        measured = steady[-1].midi - steady[0].midi
        error = abs(measured - gap)
        correct = error <= tolerance
        return ExerciseResult(
            name=f"interval {i + 1}", score=1.0 if correct else 0.0, passed=correct,
            expected=f"{gap} semitones", heard=f"{measured:.1f} semitones")

    def _exercise_tempo(self, rng: random.Random, raaga: Raaga, tolerance: float,
                        i: int) -> ExerciseResult:
        bpm = rng.choice([60, 72, 84, 96, 120])
        beat = 60.0 / bpm
        swaras = [rng.choice(raaga.allowed) for _ in range(10)]
        notes = [Note(swara=s, midi=raaga.midi(s, TONIC), start=k * beat,
                      duration=beat * 0.6) for k, s in enumerate(swaras)]
        heard = self._hear(self._render_notes(notes, "piano"), raaga)
        measured = heard.tempo_bpm
        if measured <= 0:
            return ExerciseResult(name=f"tempo {i + 1}", score=0.0, passed=False,
                                  expected=f"{bpm} bpm", heard="no pulse")
        # Octave errors (double or half time) are a near miss, not a failure.
        candidates = [measured, measured * 2, measured / 2]
        error = min(abs(c - bpm) / bpm for c in candidates)
        correct = error <= tolerance
        return ExerciseResult(
            name=f"tempo {i + 1}", score=round(max(0.0, 1.0 - error / 0.5), 3)
            if correct else 0.0,
            passed=correct, expected=f"{bpm} bpm", heard=f"{measured:.0f} bpm")

    def _listen_transcribe(self, unit: Unit, raaga: Optional[Raaga],
                           rng: random.Random, report: PracticeReport) -> None:
        raaga = raaga or self.library.require("Shankarabharanam")
        material = unit.params.get("material", "motif")
        length = int(unit.params.get("length", 5))
        round_trip = bool(unit.params.get("round_trip", False))

        for i in range(unit.exercises):
            if material == "scale":
                ascending = i % 2 == 0
                swaras = list(raaga.arohanam if ascending
                              else raaga.avarohanam)
            else:
                swaras = self._motif(raaga, rng, length)
            # Transcription happens against a drone: the Sa is given.
            audio = self._render(swaras, raaga)
            heard = self._hear(audio, raaga, fixed_tonic=float(TONIC))
            transcript = heard.swara_sequence()
            score = _sequence_match(swaras, transcript)
            detail = ""
            if round_trip and transcript:
                # Play back what was written down and check it survives.
                again = self._hear(self._render(transcript, raaga), raaga,
                                   fixed_tonic=float(TONIC))
                score = 0.6 * score + 0.4 * _sequence_match(
                    transcript, again.swara_sequence())
                detail = "round trip"
            report.exercises.append(ExerciseResult(
                name=f"transcribe {i + 1}", score=round(score, 3),
                passed=score >= 0.7, expected=" ".join(swaras),
                heard=" ".join(transcript), detail=detail))

    # ==================================================================
    # generation
    # ==================================================================
    def _motif(self, raaga: Raaga, rng: random.Random, length: int,
               bank: Optional[Sequence[Sequence[str]]] = None,
               cadence: bool = False,
               guidance: Optional[Guidance] = None) -> List[str]:
        """Invent a short line: sometimes quoting an idiom, mostly its own."""
        g = guidance or Guidance()
        tokens: List[str] = []
        current = rng.choice(raaga.graha or ["S"])
        leapt = False
        guard = 0
        while len(tokens) < length and guard < length * 8:
            guard += 1
            if bank and len(tokens) + 2 <= length and \
                    rng.random() < min(0.8, 0.35 + g.quote_more):
                fragment = list(rng.choice(list(bank)))
                # Quote at most a few notes of an idiom: the phrase has to be
                # the agent's own line, seeded by what it heard, not a copy.
                room = min(3, length - len(tokens))
                start_at = rng.randrange(max(1, len(fragment) - room + 1))
                # Built on a scratch copy first: if landing this quote would
                # replay a run the line was told to avoid, the whole quote is
                # skipped (falling through to the walk below) rather than
                # committed - but every draw above still happened, so a
                # guided attempt makes exactly the same rng calls as today.
                quoted = list(tokens)
                quote_current = current
                blocked = False
                for token in fragment[start_at:start_at + room]:
                    # Bring the quoted idiom into the octave the line is
                    # already in, so borrowing does not cause a leap.
                    token = clamp_token(raaga, token, TONIC,
                                        raaga.midi(quote_current, TONIC) - 7,
                                        raaga.midi(quote_current, TONIC) + 7)
                    if not quoted or token != quoted[-1]:
                        if g.replays(quoted + [token]):
                            blocked = True
                            break
                        quoted.append(token)
                        quote_current = token
                if blocked:
                    continue
                tokens[:] = quoted
                current = tokens[-1] if tokens else current
                leapt = False
                continue
            direction = 1 if rng.random() < 0.5 else -1
            steps = 1 if (leapt or
                          rng.random() < min(0.98, 0.85 + 0.13 * g.prefer_step)
                          ) else 2
            leapt = steps > 1
            nxt = raaga.step(current, steps * direction, direction)
            nxt = clamp_token(raaga, nxt, TONIC, TONIC - 7, TONIC + 14)
            if not g.allows_transition(current, nxt):
                nxt = clamp_token(raaga, raaga.step(current, -steps * direction,
                                                    -direction),
                                  TONIC, TONIC - 7, TONIC + 14)
                if not g.allows_transition(current, nxt):
                    continue
            # A phrase that stands still on one note is not a phrase.
            if tokens and nxt == tokens[-1]:
                nxt = clamp_token(raaga, raaga.step(current, -steps * direction,
                                                    -direction),
                                  TONIC, TONIC - 7, TONIC + 14)
                if tokens and nxt == tokens[-1]:
                    continue
            if g.vary_more and sum(1 for t in tokens
                                   if parse_swara(t)[0] == parse_swara(nxt)[0]
                                   ) >= 2:
                alt = clamp_token(raaga, raaga.step(current, -steps * direction,
                                                    -direction),
                                  TONIC, TONIC - 7, TONIC + 14)
                if g.allows_transition(current, alt) and \
                        (not tokens or alt != tokens[-1]):
                    nxt = alt
            if g.replays(tokens + [nxt]):
                replay_alt = clamp_token(
                    raaga, raaga.step(current, -steps * direction, -direction),
                    TONIC, TONIC - 7, TONIC + 14)
                if g.allows_transition(current, replay_alt) and \
                        (not tokens or replay_alt != tokens[-1]) and \
                        not g.replays(tokens + [replay_alt]):
                    nxt = replay_alt
                else:
                    continue
            current = nxt
            tokens.append(current)
        tokens = tokens[:length]
        # Any phrase of real length comes to rest; that is what makes it a phrase.
        if (cadence or g.must_end_on_nyasa or len(tokens) >= 4) and \
                raaga.nyasa and tokens:
            candidates = [s for s in raaga.nyasa
                         if parse_swara(s)[0] not in g.avoid_endings]
            if g.prefer_jeeva:
                jeeva_only = [s for s in candidates
                             if parse_swara(s)[0] in set(raaga.jeeva)]
                if jeeva_only:
                    candidates = jeeva_only
            if not candidates:
                candidates = list(raaga.nyasa)
            resting = clamp_token(raaga, rng.choice(candidates), TONIC,
                                  TONIC - 7, TONIC + 14)
            if len(tokens) > 1 and resting == tokens[-2]:
                resting = clamp_token(raaga, raaga.nyasa[0], TONIC,
                                      TONIC - 7, TONIC + 14)
            # The cadence is a note like any other: it may not make a
            # forbidden move or complete a run the line was told not to
            # replay.  Deterministic fallback, no draws.
            if len(tokens) > 1 and not self._cadence_allowed(tokens, resting, g):
                for option in candidates:
                    option = clamp_token(raaga, option, TONIC, TONIC - 7, TONIC + 14)
                    if option != tokens[-2] and \
                            self._cadence_allowed(tokens, option, g):
                        resting = option
                        break
            tokens[-1] = resting
        return tokens

    @staticmethod
    def _cadence_allowed(tokens: Sequence[str], resting: str,
                         g: Guidance) -> bool:
        return (g.allows_transition(tokens[-2], resting)
                and not g.replays(list(tokens[:-1]) + [resting]))

    def _notes_from_tokens(self, raaga: Raaga, tokens: Sequence[str],
                           rng: random.Random,
                           tempo_bpm: float = 72.0,
                           guidance: Optional[Guidance] = None) -> List[Note]:
        g = guidance or Guidance()
        notes: List[Note] = []
        expressive = set(raaga.jeeva) | set(raaga.nyasa)
        beat = 60.0 / max(30.0, tempo_bpm)
        t = 0.0
        for i, token in enumerate(tokens):
            last = i == len(tokens) - 1
            # Short notes moving, longer notes arriving: the shape of a phrase,
            # laid on the beat so it can be played with others.
            if last:
                duration = beat * rng.choice([1.5, 2.0, 2.0])
            elif i == 0:
                duration = beat * rng.choice([1.0, 1.5])
            else:
                duration = beat * rng.choice([0.5, 0.5, 1.0, 1.0, 1.5])
            base = parse_swara(token)[0]
            wants_gamaka = base in expressive or duration >= 0.7 or last
            gamaka = raaga.gamaka_for(token) if wants_gamaka else ""
            if not gamaka and wants_gamaka and duration >= 0.7:
                gamaka = "kampita"
            if g.add_gamaka and not gamaka and duration >= beat * 0.5:
                gamaka = raaga.gamaka_for(token) or "kampita"
            velocity = 96 if (i == 0 or last) else rng.randint(72, 92)
            notes.append(Note(swara=token, midi=raaga.midi(token, TONIC),
                              start=round(t, 4), duration=duration,
                              velocity=velocity, gamaka=gamaka))
            t += duration
        return notes

    def _phrase_bank(self, raaga_name: str, guidance: Guidance,
                     min_confidence: float = 0.4,
                     limit: int = 64) -> List[List[str]]:
        """The same phrases and order ``learned_phrase_bank`` would give,
        minus any this attempt was told not to quote.  With an empty
        ``guidance`` the result is identical to ``learned_phrase_bank``."""
        return [list(p.swaras) for p in
                self.repo.phrases(raaga=raaga_name, min_confidence=min_confidence,
                                  limit=limit)
                if p.id not in guidance.avoid_quoting and 2 <= len(p.swaras) <= 10]

    def _generate_pattern(self, unit: Unit, raaga: Optional[Raaga],
                          rng: random.Random, report: PracticeReport) -> None:
        raaga = raaga or self.library.require("Shankarabharanam")
        length = int(unit.params.get("length", 4))
        use_bank = bool(unit.params.get("use_learned_phrases", False))
        want_variations = int(unit.params.get("variations", 0))
        check_original = bool(unit.params.get("check_originality", False))
        cadence = bool(unit.params.get("cadence", False))
        guidance = self._guidance

        bank = self._phrase_bank(raaga.name, guidance) if use_bank else None
        index = PhraseIndex.from_repository(self.repo, raaga.name)
        evaluator = Evaluator(self.library, index)

        tempo = 72.0
        previous: List[List[str]] = []
        for i in range(unit.exercises):
            tokens = self._motif(raaga, rng, length, bank, cadence, guidance)
            notes = self._notes_from_tokens(raaga, tokens, rng, tempo, guidance)
            evaluation = evaluator.evaluate(
                notes, raaga, tonic_midi=TONIC, tempo_bpm=tempo,
                learned_phrases=bank)
            score = evaluation.overall()
            detail = "; ".join(evaluation.mistakes[:2])

            if check_original:
                originality = check_originality(tokens, index)
                if not originality.is_original:
                    score *= 0.4
                    detail = originality.summary()

            if want_variations and previous:
                # A variation must differ from what came before but stay valid.
                closest = max(_sequence_match(tokens, old) for old in previous)
                if closest > 0.85:
                    score *= 0.5
                    detail = "too close to the previous attempt"

            previous.append(tokens)
            report.artifacts.append(notes)
            report.evaluation = evaluation
            report.findings.extend(evaluation.findings)
            report.exercises.append(ExerciseResult(
                name=f"pattern {i + 1}", score=round(score, 3),
                passed=score >= unit.minimum_pass_score,
                heard=" ".join(tokens), detail=detail))

    def _generate_section(self, unit: Unit, raaga: Optional[Raaga],
                          rng: random.Random, report: PracticeReport) -> None:
        raaga = raaga or self.library.require("Shankarabharanam")
        seconds = float(unit.params.get("seconds", 20))
        kind = SectionKind(unit.params.get("section", "pallavi"))
        intensity = float(unit.params.get("intensity", 0.6))
        free_rhythm = bool(unit.params.get("free_rhythm", False))
        tempo = int(rng.choice(range(raaga.tempo_range[0], raaga.tempo_range[-1] + 1,
                                     4)) if raaga.tempo_range else 72)

        bank = learned_phrase_bank(self.repo, raaga.name)
        index = PhraseIndex.from_repository(self.repo, raaga.name)
        evaluator = Evaluator(self.library, index)

        for i in range(unit.exercises):
            section = Section(name=kind.value.title(), kind=kind, start=0.0,
                              end=seconds, intensity=intensity)
            opts = MelodyOptions(tempo_bpm=tempo, tonic_midi=TONIC,
                                 seed=rng.randint(1, 10 ** 6),
                                 duration_target=seconds, intensity=intensity)
            notes = generate_section_notes(raaga, section, opts,
                                           opts.seed, None)
            if not notes:
                report.exercises.append(ExerciseResult(
                    name=f"{kind.value} {i + 1}", score=0.0, passed=False,
                    detail="produced nothing"))
                continue
            evaluation = evaluator.evaluate(
                notes, raaga, tonic_midi=TONIC, tempo_bpm=tempo,
                expected_seconds=seconds, learned_phrases=bank,
                free_rhythm=free_rhythm)
            score = evaluation.overall()
            report.artifacts.append(notes)
            report.evaluation = evaluation
            report.findings.extend(evaluation.findings)
            report.exercises.append(ExerciseResult(
                name=f"{kind.value} {i + 1}", score=round(score, 3),
                passed=score >= unit.minimum_pass_score,
                heard=f"{len(notes)} notes over {seconds:.0f}s",
                detail="; ".join(evaluation.mistakes[:2])))

    # ==================================================================
    # recall
    # ==================================================================
    def _recall_fact(self, unit: Unit, raaga: Optional[Raaga],
                     rng: random.Random, report: PracticeReport) -> None:
        name = unit.raaga_name or (raaga.name if raaga else "")
        wanted = list(unit.params.get("facts", []))
        for key in wanted:
            fact = self.repo.best_fact(name, key)
            if fact is None:
                report.exercises.append(ExerciseResult(
                    name=f"recall {key}", score=0.0, passed=False,
                    expected=key, heard="nothing stored",
                    detail="the agent has not learned this yet"))
                continue
            score = min(1.0, fact.confidence + 0.3)
            detail = ""
            reference = self.library.get(name)
            if reference is not None:
                expected = _reference_value(reference, key)
                if expected and fact.value.split() != expected.split():
                    score *= 0.6
                    detail = f"differs from the reference: {expected}"
            report.exercises.append(ExerciseResult(
                name=f"recall {key}", score=round(score, 3),
                passed=score >= unit.minimum_pass_score * 0.9,
                expected=key, heard=fact.value, detail=detail))

    def _recall_phrases(self, unit: Unit, raaga: Optional[Raaga],
                        rng: random.Random, report: PracticeReport) -> None:
        name = unit.raaga_name or (raaga.name if raaga else "")
        wanted = int(unit.params.get("min_phrases", 6))
        min_confidence = float(unit.params.get("min_confidence", 0.4))
        phrases = self.repo.phrases(raaga=name, min_confidence=min_confidence,
                                    limit=200)
        have = len(phrases)
        score = min(1.0, have / max(1, wanted))
        detail = f"{have} phrase(s) at confidence >= {min_confidence}"
        report.exercises.append(ExerciseResult(
            name="phrases in memory", score=round(score, 3),
            passed=have >= wanted, expected=f">= {wanted}", heard=str(have),
            detail=detail))

        if unit.params.get("require_endings_on_nyasa") and raaga is not None:
            nyasa = set(raaga.nyasa)
            endings = [parse_swara(p.swaras[-1])[0] for p in phrases if p.swaras]
            share = (sum(1 for e in endings if e in nyasa) / len(endings)
                     if endings else 0.0)
            report.exercises.append(ExerciseResult(
                name="phrases rest on nyasa", score=round(min(1.0, share * 1.6), 3),
                passed=share >= 0.3, expected="phrases ending on a resting note",
                heard=f"{share:.0%}"))

        need_contours = int(unit.params.get("require_contours", 0))
        if need_contours:
            shapes = {p.contour[:1] for p in phrases if p.contour}
            score = min(1.0, len(shapes) / max(1, need_contours))
            report.exercises.append(ExerciseResult(
                name="variety of shapes", score=round(score, 3),
                passed=len(shapes) >= need_contours,
                expected=f"{need_contours} different shapes",
                heard=f"{len(shapes)}"))

    # ==================================================================
    # classification
    # ==================================================================
    def _classify_valid(self, unit: Unit, raaga: Optional[Raaga],
                        rng: random.Random, report: PracticeReport) -> None:
        raaga = raaga or self.library.require("Shankarabharanam")
        mode = unit.params.get("mode", "in_raaga_vs_out")
        bank = learned_phrase_bank(self.repo, raaga.name) or \
            [list(p) for p in raaga.prayogas]
        if not bank:
            bank = [self._motif(raaga, rng, 5) for _ in range(4)]

        neighbours = drift_neighbours(self.library, raaga)

        for i in range(unit.exercises):
            wants_valid = rng.random() < 0.5
            if wants_valid:
                tokens = list(rng.choice(bank))
                truth = "valid"
            elif mode == "neighbour_drift" and neighbours:
                other = rng.choice(neighbours)
                tokens = self._drifted(raaga, other, rng)
                truth = "invalid"
            elif mode == "which_raaga" and neighbours:
                other = rng.choice(neighbours)
                tokens = self._motif(other, rng, 5)
                truth = "invalid"
            else:
                tokens = self._corrupt(raaga, rng)
                truth = "invalid"

            answer = "valid" if self._judge_valid(raaga, tokens) else "invalid"
            correct = answer == truth
            report.exercises.append(ExerciseResult(
                name=f"judge {i + 1}", score=1.0 if correct else 0.0,
                passed=correct, expected=truth, heard=answer,
                detail=" ".join(tokens)))

    def _judge_valid(self, raaga: Raaga, tokens: Sequence[str]) -> bool:
        """The agent's own judgement, made only from what it knows."""
        allowed = set(raaga.allowed)
        bases = [parse_swara(t)[0] for t in tokens]
        if any(b not in allowed for b in bases):
            return False
        ascending_ok = set(raaga.ascending)
        descending_ok = set(raaga.descending)
        midi = [raaga.midi(t, TONIC) for t in tokens]
        for i in range(1, len(tokens)):
            if midi[i] > midi[i - 1] and bases[i] not in ascending_ok:
                return False
            if midi[i] < midi[i - 1] and bases[i] not in descending_ok:
                return False
        return True

    def _corrupt_tokens(self, raaga: Raaga, tokens: Sequence[str],
                        rng: random.Random) -> List[str]:
        """*tokens* with one note replaced by one this raaga does not have."""
        tokens = list(tokens)
        outside = [s for s in SWARA_SEMITONES if s not in set(raaga.allowed)]
        if outside:
            tokens[rng.randrange(len(tokens))] = rng.choice(outside)
        return tokens

    def _corrupt(self, raaga: Raaga, rng: random.Random) -> List[str]:
        """A phrase with a note this raaga does not have."""
        return self._corrupt_tokens(raaga, self._motif(raaga, rng, 5), rng)

    def _corrupt_direction(self, raaga: Raaga, tokens: Sequence[str],
                           rng: random.Random) -> List[str]:
        """*tokens* with one step made in a direction the raaga forbids."""
        tokens = list(tokens)
        if len(tokens) < 2:
            return tokens
        ascending_ok = set(raaga.ascending)
        descending_ok = set(raaga.descending)
        i = rng.randrange(1, len(tokens))
        prev_midi = raaga.midi(tokens[i - 1], TONIC)
        candidates = [s for s in raaga.allowed
                     if (raaga.midi(s, TONIC) > prev_midi and s not in ascending_ok)
                     or (raaga.midi(s, TONIC) < prev_midi and s not in descending_ok)]
        if candidates:
            tokens[i] = rng.choice(candidates)
        return tokens

    def _repair(self, raaga: Raaga, tokens: Sequence[str]) -> List[str]:
        """The engine's own correction: replace every offending token with
        the nearest allowed swara that keeps the move in a permitted
        direction (T7, the correction exercise)."""
        tokens = list(tokens)
        allowed = set(raaga.allowed)
        ascending_ok = set(raaga.ascending)
        descending_ok = set(raaga.descending)
        for i, token in enumerate(tokens):
            base = parse_swara(token)[0]
            offending = base not in allowed
            direction_set: Optional[set] = None
            if i > 0:
                prev_midi = raaga.midi(tokens[i - 1], TONIC)
                midi = raaga.midi(token, TONIC)
                if midi > prev_midi:
                    direction_set = ascending_ok
                elif midi < prev_midi:
                    direction_set = descending_ok
                if direction_set is not None and base not in direction_set \
                        and not offending:
                    offending = True
            if not offending:
                continue
            candidates = [s for s in (direction_set or allowed) if s in allowed]
            if not candidates:
                candidates = list(allowed)
            target_midi = raaga.midi(token, TONIC)
            best = min(candidates,
                      key=lambda s: abs(raaga.midi(s, TONIC) - target_midi))
            tokens[i] = best
        return tokens

    def _correct_phrase(self, unit: Unit, raaga: Optional[Raaga],
                        rng: random.Random, report: PracticeReport) -> None:
        """T7: given a corrupted phrase, repair it and mark how close the
        repair and the repaired line's grammar are to the original."""
        raaga = raaga or self.library.require("Shankarabharanam")
        bank = learned_phrase_bank(self.repo, raaga.name) or \
            [list(p) for p in raaga.prayogas]
        index = PhraseIndex.from_repository(self.repo, raaga.name)
        evaluator = Evaluator(self.library, index)

        for i in range(unit.exercises):
            original = list(rng.choice(bank)) if bank else \
                self._motif(raaga, rng, 5)
            if len(original) < 3:
                original = self._motif(raaga, rng, 5)
            if rng.random() < 0.5:
                corrupted = self._corrupt_tokens(raaga, original, rng)
            else:
                corrupted = self._corrupt_direction(raaga, original, rng)
            repaired = self._repair(raaga, corrupted)

            notes = self._notes_from_tokens(raaga, repaired, rng)
            evaluation = evaluator.evaluate(notes, raaga, tonic_midi=TONIC,
                                            learned_phrases=bank)
            match = _sequence_match(original, repaired)
            score = round((evaluation.scores.get("swara_correctness", 0.0)
                          + evaluation.scores.get("raaga_correctness", 0.0)
                          + match) / 3.0, 3)
            report.evaluation = evaluation
            report.findings.extend(evaluation.findings)
            report.exercises.append(ExerciseResult(
                name=f"correct {i + 1}", score=score,
                passed=score >= unit.minimum_pass_score,
                expected=" ".join(original), heard=" ".join(repaired),
                detail=f"given {' '.join(corrupted)}"))

    def _drifted(self, raaga: Raaga, neighbour: Raaga,
                 rng: random.Random) -> List[str]:
        """A phrase that slides into the neighbouring raaga."""
        tokens = self._motif(raaga, rng, 5)
        only_neighbour = [s for s in neighbour.allowed if s not in set(raaga.allowed)]
        if only_neighbour:
            tokens[-2 if len(tokens) > 2 else -1] = rng.choice(only_neighbour)
        else:
            # No exclusive note: break the direction rule instead.
            wrong = [s for s in raaga.descending if s not in set(raaga.ascending)]
            if wrong:
                tokens.insert(1, rng.choice(wrong))
        return tokens


# --------------------------------------------------------------------------
def _sequence_match(expected: Sequence[str], heard: Sequence[str]) -> float:
    """How much of the expected sequence survived, position by position."""
    if not expected:
        return 0.0
    if not heard:
        return 0.0
    import difflib
    matcher = difflib.SequenceMatcher(
        None, [parse_swara(s)[0] for s in expected],
        [parse_swara(s)[0] for s in heard])
    return round(matcher.ratio(), 4)


def _reference_value(raaga: Raaga, key: str) -> str:
    return {
        "name": raaga.name,
        "arohanam": " ".join(raaga.arohanam),
        "avarohanam": " ".join(raaga.avarohanam),
        "swaras": " ".join(raaga.allowed),
        "jeeva": " ".join(raaga.jeeva),
        "nyasa": " ".join(raaga.nyasa),
        "graha": " ".join(raaga.graha),
    }.get(key, "")
