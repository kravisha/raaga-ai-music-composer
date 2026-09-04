"""The teacher: evaluation and critique (learning specification section 14).

Scores are reported per dimension and never collapsed into one number by the
evaluator itself.  A caller that needs a single figure asks for one explicitly,
and the weighting it uses is visible here rather than hidden.

The critic also names what it thinks went wrong and what to do about it, which
is what makes a failed practice attempt useful rather than just a low score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging_setup import get_logger
from ..core.models import Note
from ..raaga.library import Raaga, RaagaLibrary, parse_swara
from ..raaga.selection import expand_feel_words
from .originality import OriginalityReport, PhraseIndex, check as check_originality

log = get_logger("agent.evaluator")

DIMENSIONS = (
    "swara_correctness",
    "raaga_correctness",
    "phrase_authenticity",
    "raaga_drift",
    "rhythm",
    "coherence",
    "originality",
    "mood_match",
    "brief_match",
    "structure",
    "interest",
    "expressiveness",
)

# Used only when a caller explicitly asks for one number.
WEIGHTS = {
    "swara_correctness": 2.0,
    "raaga_correctness": 2.0,
    "phrase_authenticity": 1.5,
    "raaga_drift": 1.5,
    "rhythm": 1.0,
    "coherence": 1.5,
    "originality": 1.5,
    "mood_match": 1.0,
    "brief_match": 1.0,
    "structure": 1.0,
    "interest": 1.0,
    "expressiveness": 1.0,
}


@dataclass
class Finding:
    """One detected mistake, with the tokens or transition that caused it, so
    a lesson can be made of it (spec section 26 "detected mistakes")."""
    dimension: str
    kind: str
    text: str
    evidence: str = ""
    weight: float = 1.0


@dataclass
class Evaluation:
    scores: Dict[str, float] = field(default_factory=dict)
    mistakes: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    confidence: float = 0.5
    recommendation: str = ""
    originality: Optional[OriginalityReport] = None
    notes_examined: int = 0

    def note(self, dimension: str, kind: str, text: str, evidence: str = "",
             weight: float = 1.0) -> None:
        """Record a mistake as both a prose string and a structured Finding."""
        self.mistakes.append(text)
        self.findings.append(Finding(dimension=dimension, kind=kind, text=text,
                                     evidence=evidence, weight=weight))

    def overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        weights = weights or WEIGHTS
        used = {k: v for k, v in self.scores.items() if k in weights}
        if not used:
            return 0.0
        total = sum(weights[k] for k in used)
        return round(sum(self.scores[k] * weights[k] for k in used) / total, 4)

    def passed(self, threshold: float) -> bool:
        return self.overall() >= threshold

    def weakest(self, count: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1])[:count]

    def report(self) -> str:
        rows = [f"overall {self.overall():.2f}  (confidence {self.confidence:.2f})"]
        for name in DIMENSIONS:
            if name in self.scores:
                rows.append(f"  {name:<20} {self.scores[name]:.2f}")
        if self.mistakes:
            rows.append("  problems:")
            rows.extend(f"    - {m}" for m in self.mistakes)
        if self.recommendation:
            rows.append(f"  next time: {self.recommendation}")
        return "\n".join(rows)


class Evaluator:
    def __init__(self, library: RaagaLibrary,
                 phrase_index: Optional[PhraseIndex] = None) -> None:
        self.library = library
        self.phrase_index = phrase_index or PhraseIndex()

    # ------------------------------------------------------------------
    def evaluate(self, notes: Sequence[Note], raaga: Raaga, *,
                 tonic_midi: int = 60,
                 brief=None,
                 tempo_bpm: float = 0.0,
                 expected_seconds: float = 0.0,
                 learned_phrases: Optional[Sequence[Sequence[str]]] = None,
                 free_rhythm: bool = False) -> Evaluation:
        evaluation = Evaluation(notes_examined=len(notes))
        if not notes:
            evaluation.scores = {d: 0.0 for d in DIMENSIONS}
            evaluation.note("structure", "nothing_produced", "nothing was produced")
            evaluation.recommendation = "generate at least a few notes"
            evaluation.confidence = 1.0
            return evaluation

        swaras = [n.swara for n in notes]
        bases = [parse_swara(s)[0] for s in swaras]
        midi = [n.midi for n in notes]

        evaluation.scores["swara_correctness"] = self._swara_correctness(
            bases, raaga, evaluation)
        evaluation.scores["raaga_correctness"] = self._raaga_correctness(
            notes, bases, raaga, evaluation)
        evaluation.scores["phrase_authenticity"] = self._authenticity(
            bases, raaga, learned_phrases, evaluation)
        evaluation.scores["raaga_drift"] = self._drift(bases, raaga, evaluation)
        evaluation.scores["rhythm"] = self._rhythm(
            notes, tempo_bpm, free_rhythm, evaluation)
        evaluation.scores["coherence"] = self._coherence(midi, evaluation)
        evaluation.scores["structure"] = self._structure(
            notes, bases, raaga, evaluation)
        evaluation.scores["interest"] = self._interest(midi, notes, evaluation)
        evaluation.scores["expressiveness"] = self._expressiveness(
            notes, evaluation)
        evaluation.scores["mood_match"] = self._mood_match(
            raaga, brief, tempo_bpm, evaluation)
        evaluation.scores["brief_match"] = self._brief_match(
            notes, brief, tempo_bpm, expected_seconds, evaluation)

        report = check_originality(swaras, self.phrase_index)
        evaluation.originality = report
        evaluation.scores["originality"] = report.score
        if not report.is_original:
            evaluation.note("originality", "not_original", report.summary(),
                            evidence=f"{report.matched_phrase_id} run "
                                     f"{report.longest_match}", weight=1.5)

        evaluation.confidence = round(
            0.5 + 0.4 * min(1.0, len(notes) / 12.0)
            + 0.1 * min(1.0, self.phrase_index.size / 20.0), 3)
        evaluation.recommendation = self._recommend(evaluation, raaga)
        return evaluation

    # -- dimensions ----------------------------------------------------
    def _swara_correctness(self, bases: Sequence[str], raaga: Raaga,
                           evaluation: Evaluation) -> float:
        allowed = set(raaga.allowed)
        wrong = [b for b in bases if b not in allowed]
        if wrong:
            evaluation.note(
                "swara_correctness", "outside_swara",
                f"{len(wrong)} note(s) outside {raaga.name}: "
                f"{', '.join(sorted(set(wrong)))}",
                evidence=" ".join(sorted(set(wrong))))
        return round(1.0 - len(wrong) / len(bases), 3)

    def _raaga_correctness(self, notes: Sequence[Note], bases: Sequence[str],
                           raaga: Raaga, evaluation: Evaluation) -> float:
        ascending_ok = set(raaga.ascending)
        descending_ok = set(raaga.descending)
        breaks = 0
        offending_transitions: List[str] = []
        for i in range(1, len(notes)):
            if notes[i].midi > notes[i - 1].midi and bases[i] not in ascending_ok:
                breaks += 1
                offending_transitions.append(f"{bases[i - 1]}>{bases[i]}")
            elif notes[i].midi < notes[i - 1].midi and bases[i] not in descending_ok:
                breaks += 1
                offending_transitions.append(f"{bases[i - 1]}>{bases[i]}")
        if breaks:
            evaluation.note(
                "raaga_correctness", "wrong_direction",
                f"{breaks} move(s) use a note the arohanam or avarohanam does "
                f"not allow in that direction",
                evidence=" ".join(offending_transitions) or str(breaks))
        forbidden = set(raaga.forbidden_swaras)
        used_forbidden = forbidden & set(bases)
        if used_forbidden:
            evaluation.note(
                "raaga_correctness", "forbidden_swara",
                f"{raaga.name} does not use {', '.join(sorted(used_forbidden))}",
                evidence=" ".join(sorted(used_forbidden)))
        penalty = breaks / max(1, len(notes) - 1) + 0.3 * len(used_forbidden)
        return round(max(0.0, 1.0 - penalty), 3)

    def _authenticity(self, bases: Sequence[str], raaga: Raaga,
                      learned: Optional[Sequence[Sequence[str]]],
                      evaluation: Evaluation) -> float:
        known: List[List[str]] = [[parse_swara(s)[0] for s in p]
                                  for p in raaga.prayogas]
        if learned:
            known.extend([parse_swara(s)[0] for s in phrase] for phrase in learned)
        if not known:
            return 0.5
        hits = 0
        for phrase in known:
            if len(phrase) < 2:
                continue
            for i in range(len(bases) - len(phrase) + 1):
                if bases[i:i + len(phrase)] == phrase:
                    hits += 1
                    break
        # Two-note idioms also count: characteristic pairs of the raaga.
        pairs = set()
        for phrase in known:
            for a, b in zip(phrase, phrase[1:]):
                pairs.add((a, b))
        pair_hits = sum(1 for a, b in zip(bases, bases[1:]) if (a, b) in pairs)
        coverage = pair_hits / max(1, len(bases) - 1)
        score = min(1.0, 0.55 * coverage + 0.45 * min(1.0, hits / 2.0))
        if score < 0.3:
            evaluation.note(
                "phrase_authenticity", "no_idiom",
                "it is in the scale but does not use the raaga's own phrases")
        elif score > 0.7:
            evaluation.strengths.append("uses characteristic phrases")
        return round(score, 3)

    def _drift(self, bases: Sequence[str], raaga: Raaga,
               evaluation: Evaluation) -> float:
        """Does another raaga explain these notes better than this one?"""
        used = set(bases)
        target = len(used & set(raaga.allowed)) / max(1, len(used))
        # Which other raaga explains these notes best?  A raaga that fits as
        # well but is *smaller* is the more dangerous neighbour, and on a tie
        # a raaga somebody curated is named ahead of a bare parent scale:
        # since the Stage 1 pack put all 72 melakartas in the library dozens
        # of scales tie here, and "whichever the library happened to yield
        # last" is not an accusation worth making.  Candidates are ranked
        # against each other; the size comparison used to be made against the
        # target raaga instead, which meant every tied candidate smaller than
        # the target replaced the last one.
        def rank(other: Raaga) -> tuple:
            return (len(used & set(other.allowed)) / max(1, len(used)),
                    -len(other.allowed), not other.scale_only)

        best_other, best_name = 0.0, ""
        best: Optional[Raaga] = None
        for other in sorted(self.library.all(), key=lambda r: r.name):
            if other.name == raaga.name:
                continue
            if best is None or rank(other) > rank(best):
                best = other
        if best is not None:
            best_other, best_name = rank(best)[0], best.name
        if best_other > target:
            evaluation.note(
                "raaga_drift", "neighbour_drift",
                f"this sounds more like {best_name} than {raaga.name}",
                evidence=best_name)
            return round(max(0.0, 1.0 - (best_other - target) * 2.0), 3)
        jeeva = set(raaga.jeeva)
        if jeeva:
            presence = sum(1 for b in bases if b in jeeva) / len(bases)
            if presence < 0.08:
                evaluation.note(
                    "raaga_drift", "neighbour_drift",
                    f"the life-giving notes of {raaga.name} "
                    f"({', '.join(raaga.jeeva)}) barely appear",
                    evidence=" ".join(raaga.jeeva))
                return 0.6
        return 1.0

    def _rhythm(self, notes: Sequence[Note], tempo_bpm: float,
                free_rhythm: bool, evaluation: Evaluation) -> float:
        if len(notes) < 2:
            return 0.5
        durations = np.array([n.duration for n in notes], dtype=np.float32)
        if np.any(durations <= 0):
            evaluation.note("rhythm", "off_beat", "some notes have no length",
                            evidence=str(int(np.sum(durations <= 0))))
            return 0.0
        if free_rhythm:
            # Alapana: reward long, varied, unhurried notes instead of a grid.
            variety = float(np.std(durations) / max(1e-6, np.mean(durations)))
            return round(min(1.0, 0.45 + variety), 3)
        if tempo_bpm <= 0:
            return 0.6
        beat = 60.0 / tempo_bpm
        ratios = durations / beat
        offsets = np.abs(ratios - np.round(ratios * 2) / 2)   # half-beat grid
        score = float(np.clip(1.0 - offsets.mean() * 3.0, 0.0, 1.0))
        if score < 0.5:
            evaluation.note("rhythm", "off_beat",
                            "the note lengths do not sit on the beat",
                            evidence=f"{float(offsets.mean()):.3f}")
        return round(score, 3)

    def _coherence(self, midi: Sequence[float], evaluation: Evaluation) -> float:
        if len(midi) < 3:
            return 0.5
        steps = np.abs(np.diff(np.array(midi, dtype=np.float32)))
        leaps = float(np.mean(steps > 7))
        stepwise = float(np.mean(steps <= 2))
        score = 0.6 * stepwise + 0.4 * (1.0 - leaps)
        if leaps > 0.3:
            evaluation.note("coherence", "too_many_leaps",
                            "too many large jumps; the line is disjointed",
                            evidence=str(int(np.sum(steps > 7))))
        elif stepwise > 0.55:
            evaluation.strengths.append("the line moves smoothly")
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def _structure(self, notes: Sequence[Note], bases: Sequence[str],
                   raaga: Raaga, evaluation: Evaluation) -> float:
        if len(notes) < 3:
            return 0.3
        score = 0.4
        if raaga.nyasa and bases[-1] in set(raaga.nyasa):
            score += 0.35
        else:
            evaluation.note(
                "structure", "no_cadence",
                f"it does not come to rest on a resting note "
                f"({', '.join(raaga.nyasa) or 'none defined'})",
                evidence=bases[-1])
        if raaga.graha and bases[0] in set(raaga.graha):
            score += 0.1
        midi = [n.midi for n in notes]
        first_third = midi[:max(1, len(midi) // 3)]
        middle = midi[max(1, len(midi) // 3):max(2, 2 * len(midi) // 3)]
        if middle and max(middle) > max(first_third):
            score += 0.15                     # it goes somewhere before returning
        return round(min(1.0, score), 3)

    def _interest(self, midi: Sequence[float], notes: Sequence[Note],
                  evaluation: Evaluation) -> float:
        if len(midi) < 3:
            return 0.3
        distinct = len(set(int(m) for m in midi)) / len(midi)
        durations = [n.duration for n in notes]
        rhythmic_variety = len(set(round(d, 2) for d in durations)) / len(durations)
        span = (max(midi) - min(midi)) / 12.0
        score = 0.4 * distinct + 0.3 * rhythmic_variety + 0.3 * min(1.0, span)
        if distinct < 0.3:
            evaluation.note("interest", "repetitive", "it repeats the same few notes")
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def _expressiveness(self, notes: Sequence[Note],
                        evaluation: Evaluation) -> float:
        gamaka = sum(1 for n in notes if n.gamaka) / len(notes)
        velocities = [n.velocity for n in notes]
        dynamic = (max(velocities) - min(velocities)) / 60.0
        long_notes = sum(1 for n in notes if n.duration > 0.8) / len(notes)
        score = 0.45 * min(1.0, gamaka * 3.0) + 0.3 * min(1.0, dynamic) \
            + 0.25 * min(1.0, long_notes * 3.0)
        if gamaka == 0:
            evaluation.note("expressiveness", "no_gamaka",
                            "no ornamentation at all; it sounds flat")
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    def _mood_match(self, raaga: Raaga, brief, tempo_bpm: float,
                    evaluation: Evaluation) -> float:
        if brief is None:
            return 0.6
        words = expand_feel_words(getattr(brief, "mood", ""),
                                  getattr(brief, "feel", ""),
                                  getattr(brief, "situation", ""))
        if not words:
            return 0.6
        overlap = len(set(words) & set(raaga.moods)) / max(1, len(set(raaga.moods)))
        score = min(1.0, 0.35 + overlap * 2.0)
        if tempo_bpm and raaga.tempo_range:
            low, high = raaga.tempo_range[0], raaga.tempo_range[-1]
            if not (low - 10 <= tempo_bpm <= high + 10):
                score *= 0.8
                evaluation.note(
                    "mood_match", "mood_mismatch",
                    f"{tempo_bpm:.0f} bpm sits outside the comfortable range for "
                    f"{raaga.name} ({low}-{high})",
                    evidence=f"{tempo_bpm:.0f}")
        if score < 0.4:
            evaluation.note(
                "mood_match", "mood_mismatch",
                f"{raaga.name} does not obviously carry "
                f"{', '.join(sorted(set(words))[:3])}",
                evidence=" ".join(sorted(set(words))[:3]))
        return round(score, 3)

    def _brief_match(self, notes: Sequence[Note], brief, tempo_bpm: float,
                     expected_seconds: float, evaluation: Evaluation) -> float:
        score = 0.7
        if expected_seconds > 0:
            actual = max(n.start + n.duration for n in notes)
            ratio = actual / expected_seconds
            if 0.75 <= ratio <= 1.3:
                score += 0.3
            else:
                evaluation.note(
                    "brief_match", "brief_mismatch",
                    f"asked for about {expected_seconds:.0f}s, produced "
                    f"{actual:.0f}s", evidence="length")
                score -= 0.3
        if brief is not None:
            wanted = getattr(brief, "tempo_preference", None)
            if wanted and tempo_bpm and abs(wanted - tempo_bpm) > 12:
                evaluation.note(
                    "brief_match", "brief_mismatch",
                    f"asked for {wanted} bpm, produced {tempo_bpm:.0f}",
                    evidence="tempo")
                score -= 0.2
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    # -- advice --------------------------------------------------------
    @staticmethod
    def _recommend(evaluation: Evaluation, raaga: Raaga) -> str:
        weakest = evaluation.weakest(1)
        if not weakest:
            return ""
        name, value = weakest[0]
        if value > 0.75:
            return "nothing pressing; refine the phrasing"
        advice = {
            "swara_correctness": f"stay inside {raaga.name}: "
                                 f"{', '.join(raaga.allowed)}",
            "raaga_correctness": "respect the arohanam going up and the "
                                 "avarohanam coming down",
            "phrase_authenticity": f"quote the raaga's own phrases, such as "
                                   f"{' '.join(raaga.prayogas[0]) if raaga.prayogas else 'its prayogas'}",
            "raaga_drift": f"lean on {', '.join(raaga.jeeva) or 'the jeeva swaras'} "
                           f"so the raaga stays recognisable",
            "rhythm": "line the note lengths up with the beat",
            "coherence": "move more by step and save the leaps",
            "originality": "invent a new line instead of quoting one that was learned",
            "mood_match": "choose a raaga or tempo closer to the mood asked for",
            "brief_match": "match the requested length and tempo",
            "structure": f"end on a resting note "
                         f"({', '.join(raaga.nyasa) or 'a nyasa swara'})",
            "interest": "vary the notes and the rhythm more",
            "expressiveness": "add gamaka and let some notes breathe",
        }
        return advice.get(name, f"work on {name}")


def evaluate_notes(notes: Sequence[Note], raaga: Raaga,
                   library: RaagaLibrary, **kwargs) -> Evaluation:
    return Evaluator(library).evaluate(notes, raaga, **kwargs)
