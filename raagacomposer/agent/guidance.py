"""Guidance: what the agent's own lessons tell it before the next attempt.

The Retrieval / Context Builder of the learning specification (section 15,
component 9) applied to practice.  Lessons (section 38) are stored facts
about what went wrong; guidance is what a generator can obey while writing
the next line.  It constrains, it never supplies the answer: a lesson says
"the line ended on Ri, which is not a resting note", the guidance says "end
on a resting note", and the generator still chooses which one.  The
evaluator is never shown the guidance, so a pass is still the line's own.

Everything here is a pure function of the knowledge base.  The same lessons
give the same guidance in every process, and no guidance at all leaves the
generators exactly as they were (docs/PLAN_learning_loop.md, "Determinism
stays").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..raaga.library import parse_swara
from .knowledge import KnowledgeRepository, Lesson

# A lesson learned on the unit being attempted weighs this much more than
# the same lesson learned elsewhere in the raaga.
UNIT_WEIGHT = 2.0
# The most any one lesson kind can push a probability.
MAX_STEP_BIAS = 0.5
MAX_QUOTE_BIAS = 0.4
# A line replays a forbidden phrase once this many of its notes in a row
# match it.  Three: the same limit practice puts on deliberate quotation
# (docs/DECISIONS.md, "Originality is enforced").
REPLAY_WINDOW = 3


@dataclass
class Guidance:
    """Constraints for the next attempt, derived from lessons.

    ``avoid_transitions`` holds (from, to) pairs of base swaras the line must
    not make; ``avoid_swaras`` base swaras it must not use; ``avoid_endings``
    base swaras it must not end on; ``must_end_on_nyasa`` forces a cadence;
    ``avoid_quoting`` holds ids of learned phrases it must not quote;
    ``prefer_step`` (0..1) and ``quote_more`` (0..1) shift the generator's
    existing probabilities; ``prefer_jeeva`` asks cadences to land on the
    raaga's life-giving notes; ``vary_more`` and ``add_gamaka`` are flags
    the note-shaping stage reads.  ``lesson_ids`` records which lessons
    produced this guidance so they can be marked applied when the attempt
    passes, and ``kinds`` names them for a log line.
    """
    avoid_transitions: Set[Tuple[str, str]] = field(default_factory=set)
    avoid_swaras: Set[str] = field(default_factory=set)
    avoid_endings: Set[str] = field(default_factory=set)
    must_end_on_nyasa: bool = False
    avoid_quoting: Set[str] = field(default_factory=set)
    # Base-swara sequences the line must not replay: the learned phrases it
    # was caught copying.  ``avoid_quoting`` stops deliberate quotation;
    # this stops the free walk from wandering into the same run by accident.
    avoid_runs: Set[Tuple[str, ...]] = field(default_factory=set)
    prefer_step: float = 0.0
    quote_more: float = 0.0
    prefer_jeeva: bool = False
    vary_more: bool = False
    add_gamaka: bool = False
    lesson_ids: List[str] = field(default_factory=list)
    kinds: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.avoid_transitions or self.avoid_swaras
                    or self.avoid_endings or self.must_end_on_nyasa
                    or self.avoid_quoting or self.avoid_runs or self.prefer_step
                    or self.quote_more or self.prefer_jeeva
                    or self.vary_more or self.add_gamaka)

    def describe(self) -> str:
        """One line for a learning step's detail: what it was told."""
        if self.is_empty():
            return ""
        parts: List[str] = []
        if self.avoid_swaras:
            parts.append("avoid " + " ".join(sorted(self.avoid_swaras)))
        if self.avoid_transitions:
            parts.append("never " + " ".join(
                f"{a}>{b}" for a, b in sorted(self.avoid_transitions)))
        if self.must_end_on_nyasa:
            parts.append("end on a resting note")
        elif self.avoid_endings:
            parts.append("do not end on " + " ".join(sorted(self.avoid_endings)))
        if self.prefer_step:
            parts.append(f"move by step ({self.prefer_step:.2f})")
        if self.avoid_quoting or self.avoid_runs:
            parts.append(f"do not replay "
                         f"{max(len(self.avoid_quoting), len(self.avoid_runs))}"
                         f" learned phrase(s)")
        if self.quote_more:
            parts.append(f"quote the idiom more ({self.quote_more:.2f})")
        if self.prefer_jeeva:
            parts.append("rest on the jeeva swaras")
        if self.vary_more:
            parts.append("vary the notes")
        if self.add_gamaka:
            parts.append("add gamaka")
        return "; ".join(parts)

    def allows_transition(self, a: str, b: str) -> bool:
        pair = (parse_swara(a)[0], parse_swara(b)[0])
        return pair not in self.avoid_transitions and pair[1] not in self.avoid_swaras

    def allows_ending(self, token: str) -> bool:
        return parse_swara(token)[0] not in self.avoid_endings

    def replays(self, tokens: Sequence[str], window: int = REPLAY_WINDOW) -> bool:
        """Would the last ``window`` tokens be a run copied from a phrase the
        line was told not to replay?  Octave marks are ignored, as the
        originality checker ignores them."""
        if not self.avoid_runs or len(tokens) < window:
            return False
        tail = tuple(parse_swara(t)[0] for t in tokens[-window:])
        for run in self.avoid_runs:
            if len(run) < window:
                continue
            for start in range(len(run) - window + 1):
                if run[start:start + window] == tail:
                    return True
        return False


def _tokens(evidence: str) -> List[str]:
    return [t for t in (evidence or "").replace(",", " ").split() if t]


def guidance_from_lessons(lessons: Sequence[Lesson], unit_id: str = "",
                          phrases: Optional[Dict[str, Sequence[str]]] = None
                          ) -> Guidance:
    """Fold lessons into one Guidance.  Deterministic: the order of the
    lessons does not matter, only their kinds, evidence and weights.

    ``phrases`` maps phrase ids to their swaras, for the ``not_original``
    lessons whose ``related`` names the phrase that was copied; without it
    those lessons forbid quoting the phrase but cannot forbid replaying it.
    """
    guidance = Guidance()
    phrases = phrases or {}
    step_bias = 0.0
    quote_bias = 0.0
    kinds: Dict[str, float] = {}
    for lesson in sorted(lessons, key=lambda l: (l.kind, l.id)):
        if lesson.applied:
            continue
        weight = float(max(1, lesson.recurrences))
        if unit_id and lesson.unit_id == unit_id:
            weight *= UNIT_WEIGHT
        kind = lesson.kind
        if kind in ("outside_swara", "forbidden_swara"):
            guidance.avoid_swaras.update(parse_swara(t)[0] for t in _tokens(lesson.evidence))
        elif kind == "wrong_direction":
            for pair in _tokens(lesson.evidence):
                a, sep, b = pair.partition(">")
                if sep and a and b:
                    guidance.avoid_transitions.add((parse_swara(a)[0], parse_swara(b)[0]))
        elif kind == "no_cadence":
            guidance.must_end_on_nyasa = True
            for t in _tokens(lesson.evidence):
                guidance.avoid_endings.add(parse_swara(t)[0])
        elif kind == "too_many_leaps":
            step_bias += 0.15 * weight
        elif kind == "not_original":
            guidance.avoid_quoting.update(lesson.related)
            for phrase_id in lesson.related:
                swaras = phrases.get(phrase_id)
                if swaras:
                    guidance.avoid_runs.add(
                        tuple(parse_swara(s)[0] for s in swaras))
        elif kind == "no_idiom":
            quote_bias += 0.1 * weight
        elif kind == "repetitive":
            guidance.vary_more = True
        elif kind == "no_gamaka":
            guidance.add_gamaka = True
        elif kind == "neighbour_drift":
            guidance.prefer_jeeva = True
        else:
            # Listening and recall lessons, creator feedback without a
            # finding, and kinds no generator can act on: recorded, shown
            # under LEARN, but nothing to obey here.
            continue
        kinds[kind] = kinds.get(kind, 0.0) + weight
        guidance.lesson_ids.append(lesson.id)
    guidance.prefer_step = round(min(MAX_STEP_BIAS, step_bias), 3)
    guidance.quote_more = round(min(MAX_QUOTE_BIAS, quote_bias), 3)
    guidance.kinds = sorted(kinds, key=lambda k: (-kinds[k], k))
    return guidance


def build_guidance(repo: KnowledgeRepository, raaga: str, unit_id: str = "",
                   limit: int = 200) -> Guidance:
    """Guidance for the next attempt at ``unit_id`` in ``raaga`` from the
    lessons on record.  Lessons of the unit itself count double; lessons
    from the raaga's other units and from compositions count once."""
    if not raaga:
        return Guidance()
    lessons = repo.lessons(raaga=raaga, limit=limit)
    phrases: Dict[str, Sequence[str]] = {}
    for lesson in lessons:
        if lesson.kind != "not_original":
            continue
        for phrase_id in lesson.related:
            if phrase_id not in phrases:
                phrase = repo.phrase(phrase_id)
                if phrase is not None:
                    phrases[phrase_id] = list(phrase.swaras)
    return guidance_from_lessons(lessons, unit_id=unit_id, phrases=phrases)
