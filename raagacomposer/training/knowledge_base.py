"""Phase G - storing what survived, and letting it reach the music.

Specification section 9, and section 20 rule 12: the training feature must
connect directly to the music-generation knowledge base.  That rule is the
whole point of the exercise.  A training system that fills its own private
table and never changes a note the application plays has taught nobody
anything, so this module does two things rather than one:

1. writes every accepted entry into the training knowledge base, with the
   provenance section 16 demands;
2. passes the subset that is genuinely *musical evidence* through to the
   agent's own :class:`KnowledgeRepository`, where ``learned_raaga()`` rebuilds
   the raaga from it and the composer plays what was learned.

The second step is deliberately narrower than the first, and the line is one
of evidence rather than of confidence.  A phrase the system *heard* - pitch
tracked, swaras identified, timestamp recorded - is an observation, and it may
reach the composer.  A phrase a teacher merely *stated* in a transcript has
not been verified by ear, and it stays in the training record where a person
can see it, rather than becoming something the application performs.  Storing
both and playing only one is what keeps "we read that this is a Kambhoji
phrase" from turning into "this is a Kambhoji phrase".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..raaga.library import RaagaLibrary, parse_swara
from .models import KnowledgeEntry, KnowledgeStatus, LearningSource
from .store import TrainingStore

log = get_logger("training.knowledge_base")

#: Only phrases carrying this tag were actually listened to.
HEARD = "heard"


def _kb_type(category: str) -> str:
    """A training category, as a Knowledge Base knowledge type."""
    from ..kb.models import KnowledgeType

    return {
        "phrase": KnowledgeType.PATTERN,
        "grammar": KnowledgeType.CONSTRAINT,
        "practice": KnowledgeType.PROCEDURE,
        "self-assessment": KnowledgeType.META,
    }.get(category, KnowledgeType.FACT)


class KnowledgeBaseService:
    """Section 9 storage, plus the bridge to what the application plays."""

    def __init__(self, store: TrainingStore, raagas: RaagaLibrary,
                 agent_repo=None, kb=None) -> None:
        self.store = store
        self.raagas = raagas
        #: The agent's own memory.  Optional: training still works without it,
        #: it simply cannot change what the composer plays.
        self.agent_repo = agent_repo
        #: The durable Knowledge Base.  The training store keeps the record of
        #: *this run*; the Knowledge Base accumulates across all of them, which
        #: is the distinction the knowledge-base specification draws in its
        #: section 26.  Optional, so training still works without one.
        self.kb = kb

    # ------------------------------------------------------------------
    def store_all(self, entries: Sequence[KnowledgeEntry],
                  source: LearningSource, run_id: str) -> List[str]:
        """Write accepted knowledge, and pass musical evidence to the agent."""
        stored: List[str] = []
        for entry in entries:
            if not entry.source_id or not entry.run_id:
                # Section 16: an entry that cannot say where it came from has
                # no business in the store.
                log.warning("refusing an entry with no provenance: %s",
                            entry.normalized_statement[:80])
                continue
            self.store.add_knowledge(entry)
            stored.append(entry.knowledge_id)
        self._bridge_to_agent(entries, source, run_id)
        self._bridge_to_kb(entries, source, run_id)
        return stored

    # ------------------------------------------------------------------
    def _bridge_to_kb(self, entries: Sequence[KnowledgeEntry],
                      source: LearningSource, run_id: str) -> int:
        """Integrate this run's findings into the durable Knowledge Base.

        The Learning Report says what happened once; the Knowledge Base is
        where it becomes part of what the application knows.  Everything goes
        through ``commit_knowledge``, so a second source teaching the same
        thing attaches evidence rather than making another row, and one
        teaching something different produces a recorded conflict rather than
        an overwrite.
        """
        if self.kb is None:
            return 0
        try:
            from ..kb.models import (Evidence, ExtractionMethod, KnowledgeItem,
                                     KnowledgeType, Scope, Source as KBSource)
            from ..kb import normalize as kb_normalize
        except Exception:  # noqa: BLE001
            return 0

        try:
            kb_source = self.kb.add_source(KBSource(
                source_type=source.source_type or "training",
                title=source.title,
                author_or_channel=source.author,
                reference=source.url,
                training_source_id=source.source_id,
                language=source.language,
                license_or_access_notes=str(
                    source.metadata.get("rights", "")) or "",
                metadata={"provider": source.provider}))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not register the source with the Knowledge "
                        "Base: %s", exc)
            return 0

        written = 0
        for entry in entries:
            try:
                predicate = kb_normalize.normalise_predicate(
                    entry.concept or entry.category or "note")
                heard = HEARD in entry.tags or "measured" in entry.tags
                item = KnowledgeItem(
                    canonical_name=entry.subject or entry.raga,
                    knowledge_type=_kb_type(entry.category),
                    subject=entry.subject or entry.raga,
                    predicate=predicate,
                    object_value=entry.normalized_statement,
                    statement=entry.normalized_statement,
                    structured_value=kb_normalize.structured_for(
                        predicate, entry.normalized_statement),
                    scope=[Scope.CARNATIC, Scope.TRAINING],
                    raga=entry.raga, tala=entry.tala,
                    difficulty=entry.difficulty, tags=list(entry.tags),
                    learned_by="training", language=source.language)
                evidence = Evidence(
                    source_id=kb_source.source_id,
                    source_segment=entry.source_timestamp,
                    transcript_excerpt=entry.evidence,
                    strength=entry.confidence,
                    extraction_method=(ExtractionMethod.AUDIO if heard
                                       else ExtractionMethod.TRANSCRIPT
                                       if "stated" in entry.tags
                                       else ExtractionMethod.INFERRED),
                    run_id=run_id)
                result = self.kb.commit_knowledge(
                    item, [evidence],
                    source_quality=0.8 if heard else 0.65, run_id=run_id)
                if result.stored:
                    written += 1
            except Exception as exc:  # noqa: BLE001 - one item is not the run
                log.debug("could not integrate an item into the Knowledge "
                          "Base: %s", exc)
        if written:
            self.store.audit(
                "knowledge.integrated",
                f"{written} item(s) integrated into the Knowledge Base",
                run_id=run_id, source_id=source.source_id)
        return written

    # ------------------------------------------------------------------
    def _bridge_to_agent(self, entries: Sequence[KnowledgeEntry],
                         source: LearningSource, run_id: str) -> int:
        """Give the composer the phrases that were genuinely heard."""
        if self.agent_repo is None:
            return 0
        try:
            from ..agent.knowledge import Phrase, Source as AgentSource
        except Exception:  # noqa: BLE001
            return 0

        phrases = [e for e in entries
                   if e.category == "phrase" and HEARD in e.tags
                   and e.status == KnowledgeStatus.ACTIVE and e.raga]
        if not phrases:
            return 0

        rights = str(source.metadata.get("rights", "")) or (
            "internally-generated" if source.source_type == "exercise"
            else "user-supplied")
        try:
            agent_source, _ = self.agent_repo.add_source(AgentSource(
                locator=source.url or source.source_id,
                title=f"training: {source.title}",
                performer=source.author, raaga=phrases[0].raga,
                content_type="audio", rights_status=rights,
                provider="training", quality=0.7,
                notes=f"learned through the Training tab, run {run_id}"))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not register the training source with the "
                        "agent: %s", exc)
            return 0

        written = 0
        for entry in phrases:
            swaras = self._swaras_of(entry)
            if len(swaras) < 3:
                continue
            raaga = self.raagas.get(entry.raga)
            if raaga is None:
                continue
            allowed = set(raaga.allowed)
            if any(parse_swara(s)[0] not in allowed for s in swaras):
                # Heard, but not in this raaga.  Interesting for the report;
                # not something to hand the composer as an idiom.
                continue
            try:
                _, is_new = self.agent_repo.add_phrase(Phrase(
                    raaga=entry.raga, swaras=swaras,
                    midi=[raaga.midi(s, 60) for s in swaras],
                    durations=[0.4] * len(swaras),
                    function="phrase", source_id=agent_source.id,
                    confidence=entry.confidence,
                    notes=f"heard at {entry.source_timestamp} in "
                          f"{source.title}"))
            except Exception as exc:  # noqa: BLE001
                log.debug("could not store a phrase with the agent: %s", exc)
                continue
            written += int(bool(is_new))

        if written:
            self.store.audit(
                "knowledge.bridged",
                f"{written} heard phrase(s) passed to the composer",
                run_id=run_id, source_id=source.source_id)
            log.info("%d phrase(s) from %s are now available to the composer",
                     written, source.title)
        return written

    @staticmethod
    def _swaras_of(entry: KnowledgeEntry) -> List[str]:
        """Recover the swaras from a phrase statement."""
        text = entry.normalized_statement
        if ":" in text:
            text = text.split(":", 1)[1]
        tokens = [t for t in text.replace(".", " ").split() if t]
        out: List[str] = []
        for token in tokens:
            base = token.strip(",.;")
            if not base:
                continue
            head = base.rstrip("+-")
            if head and head[0] in "SRGMPDN" and len(head) <= 2:
                out.append(base)
        return out

    # ------------------------------------------------------------------
    # searching (section 9)
    # ------------------------------------------------------------------
    def search(self, **criteria) -> List[KnowledgeEntry]:
        return self.store.search_knowledge(**criteria)

    def entry(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        return self.store.knowledge(knowledge_id)

    def provenance(self, knowledge_id: str) -> Dict[str, Any]:
        """Section 16 - every question it asks, answered for one item."""
        entry = self.store.knowledge(knowledge_id)
        if entry is None:
            return {}
        run = self.store.run(entry.run_id) if entry.run_id else None
        objectives = self.store.objectives(entry.run_id) if entry.run_id else []
        objective = next((o for o in objectives
                          if o.objective_id == entry.objective_id), None)
        return {
            "where_from": entry.source_title or entry.source_id,
            "source_id": entry.source_id,
            "source_url": entry.source_url,
            "learning_run": entry.run_id,
            "search_phrase": run.search_phrase if run else "",
            "objective": objective.description if objective else "",
            "timestamp": entry.source_timestamp,
            "evidence": entry.evidence,
            "confidence": entry.confidence,
            "user_approved": entry.user_approved,
            "ever_contradicted": entry.contradicted,
            "status": entry.status,
            "audit": self.store.audit_trail(knowledge_id=knowledge_id),
        }

    # -- the creator's verdicts (section 13) ------------------------------
    def mark_incorrect(self, knowledge_id: str, reason: str = "") -> None:
        self.store.update_knowledge(knowledge_id,
                                    status=KnowledgeStatus.REJECTED,
                                    contradicted=True)
        self.store.audit("knowledge.rejected", reason or "marked incorrect",
                         knowledge_id=knowledge_id)

    def approve(self, knowledge_id: str, note: str = "") -> None:
        entry = self.store.knowledge(knowledge_id)
        if entry is None:
            return
        self.store.update_knowledge(knowledge_id, user_approved=True,
                                    status=KnowledgeStatus.ACTIVE)
        self.store.audit("knowledge.approved", note or "approved by the user",
                         knowledge_id=knowledge_id)

    def count(self) -> int:
        return self.store.knowledge_count()
