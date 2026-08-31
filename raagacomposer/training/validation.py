"""Phase F - deciding what may be believed.

Specification section 7F and section 8.7.  Nothing an observation says gets
into permanent storage without passing here, and the rules are the ones the
specification is most insistent about:

* uncertain inference does not become fact.  An observation below the
  confidence floor is reported as uncertain and left out of the knowledge
  base rather than written down slightly hedged.
* a conflict is never resolved by overwriting.  The existing claim stays, the
  new claim is recorded beside it with its evidence, and a recommendation is
  offered to a person.  Section 20 rule 8 allows nothing else.
* alternative interpretations are preserved.  Two sources disagreeing is a
  fact about the material, not a bug to be tidied away.

The confidence floor is deliberately low.  Its job is not to admit only what
is certain - almost nothing about hearing music is - but to stop the pipeline
laundering a guess into a record that later looks authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from .models import (Conflict, KnowledgeEntry, KnowledgeStatus,
                     LearningSource)
from .semantics import Observation
from .store import TrainingStore

log = get_logger("training.validation")

#: Below this, an observation is reported but not stored as knowledge.
CONFIDENCE_FLOOR = 0.30
#: A new claim must beat an existing one by this much before it is even
#: *recommended* - anything closer is a genuine disagreement, not an update.
DECISIVE_MARGIN = 0.25


@dataclass
class ValidationOutcome:
    """What survived, what collided, and what was not good enough."""

    accepted: List[KnowledgeEntry] = field(default_factory=list)
    confirmed: List[KnowledgeEntry] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    rejected: List[Tuple[Observation, str]] = field(default_factory=list)
    uncertain: List[Observation] = field(default_factory=list)

    @property
    def anything_learned(self) -> bool:
        return bool(self.accepted)


class KnowledgeValidationService:
    """Section 7F: consistency, comparison, conflict, confidence."""

    def __init__(self, store: TrainingStore,
                 confidence_floor: float = CONFIDENCE_FLOOR) -> None:
        self.store = store
        self.confidence_floor = confidence_floor

    # ------------------------------------------------------------------
    def validate(self, observations: Sequence[Observation],
                 source: LearningSource, run_id: str,
                 objective_ids: Optional[Dict[str, str]] = None,
                 source_quality: float = 0.6) -> ValidationOutcome:
        outcome = ValidationOutcome()
        objective_ids = objective_ids or {}

        for observation, reason in self._inconsistent(observations):
            outcome.rejected.append((observation, reason))
        inconsistent = {id(o) for o, _ in outcome.rejected}

        for observation in observations:
            if id(observation) in inconsistent:
                continue

            confidence = self._confidence(observation, source_quality)
            if confidence < self.confidence_floor:
                observation.confidence = confidence
                outcome.uncertain.append(observation)
                continue

            entry = self._to_entry(observation, source, run_id, confidence,
                                   objective_ids)
            existing = self.store.existing_knowledge(entry)

            if existing is None:
                outcome.accepted.append(entry)
                continue

            if self._same_claim(existing, entry):
                # Independent reinforcement.  The stored entry gains a little
                # confidence, capped, and the source is credited in the report.
                raised = round(min(0.99, existing.confidence
                                   + 0.08 * entry.confidence), 3)
                if raised > existing.confidence:
                    self.store.update_knowledge(existing.knowledge_id,
                                                confidence=raised)
                    existing.confidence = raised
                outcome.confirmed.append(existing)
                continue

            # A real disagreement.  Both survive; a person decides.
            conflict = self._conflict(existing, entry, run_id)
            outcome.conflicts.append(conflict)
            entry.status = KnowledgeStatus.DISPUTED
            entry.contradicted = True
            outcome.accepted.append(entry)

        log.info("validation for %s: %d accepted, %d confirmed, %d conflict(s),"
                 " %d uncertain, %d rejected", source.title,
                 len(outcome.accepted), len(outcome.confirmed),
                 len(outcome.conflicts), len(outcome.uncertain),
                 len(outcome.rejected))
        return outcome

    # ------------------------------------------------------------------
    @staticmethod
    def _inconsistent(observations: Sequence[Observation]
                      ) -> List[Tuple[Observation, str]]:
        """Internal consistency: does this source contradict itself?"""
        bad: List[Tuple[Observation, str]] = []
        by_concept: Dict[Tuple[str, str], List[Observation]] = {}
        for observation in observations:
            if observation.concept in ("phrase", "swara sequence"):
                continue        # many phrases per source is normal
            key = (observation.raga, observation.concept)
            by_concept.setdefault(key, []).append(observation)

        for (raga, concept), group in by_concept.items():
            if len(group) < 2:
                continue
            statements = {o.statement for o in group}
            if len(statements) > 1:
                # Keep the best-supported one; the rest are the source
                # disagreeing with itself and are not written down.
                best = max(group, key=lambda o: o.confidence)
                for observation in group:
                    if observation is not best:
                        bad.append((
                            observation,
                            f"the source says more than one thing about "
                            f"{concept}{' in ' + raga if raga else ''}; the "
                            f"better-supported statement was kept"))
        return bad

    def _confidence(self, observation: Observation,
                    source_quality: float) -> float:
        """What the observation earned, tempered by where it came from."""
        confidence = float(observation.confidence)
        if "stated" in observation.tags:
            # Somebody saying a thing is weaker evidence than hearing it.
            confidence *= 0.85
        if "disagreement" in observation.tags:
            confidence *= 0.7
        confidence = 0.75 * confidence + 0.25 * (confidence * source_quality)
        return round(max(0.0, min(1.0, confidence)), 3)

    @staticmethod
    def _to_entry(observation: Observation, source: LearningSource,
                  run_id: str, confidence: float,
                  objective_ids: Dict[str, str]) -> KnowledgeEntry:
        return KnowledgeEntry(
            subject=observation.subject or source.title,
            concept=observation.concept,
            normalized_statement=observation.statement,
            category=observation.category,
            raga=observation.raga, tala=observation.tala,
            difficulty=observation.difficulty
            or str(source.metadata.get("difficulty", "")),
            source_id=source.source_id, source_url=source.url,
            source_title=source.title,
            source_timestamp=observation.reference,
            evidence=observation.evidence, confidence=confidence,
            tags=list(observation.tags), run_id=run_id,
            objective_id=objective_ids.get(observation.objective_category, ""),
            status=KnowledgeStatus.ACTIVE)

    @staticmethod
    def _same_claim(existing: KnowledgeEntry, incoming: KnowledgeEntry) -> bool:
        a = existing.normalized_statement.strip().lower().rstrip(".")
        b = incoming.normalized_statement.strip().lower().rstrip(".")
        return a == b

    def _conflict(self, existing: KnowledgeEntry, incoming: KnowledgeEntry,
                  run_id: str) -> Conflict:
        difference = incoming.confidence - existing.confidence
        if difference > DECISIVE_MARGIN:
            recommendation = (
                "the new source is markedly better supported; consider "
                "approving it and superseding what is held")
        elif difference < -DECISIVE_MARGIN:
            recommendation = (
                "what is already held is better supported; consider keeping "
                "it and recording the new claim as an alternative")
        else:
            recommendation = (
                "the two are about equally supported; both have been kept and "
                "a person should decide - teachers do differ here")
        conflict = Conflict(
            run_id=run_id, knowledge_id=existing.knowledge_id,
            existing_claim=existing.normalized_statement,
            new_claim=incoming.normalized_statement,
            source_evidence=incoming.evidence,
            existing_confidence=existing.confidence,
            new_confidence=incoming.confidence,
            recommendation=recommendation)
        # The existing entry is flagged, never rewritten.
        self.store.update_knowledge(existing.knowledge_id, contradicted=True)
        return conflict
