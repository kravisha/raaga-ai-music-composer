"""What the ears heard, written into the permanent Knowledge Base.

The agent has two memories and until now only one of them heard anything.

``knowledge.db`` is the agent's own repository: phrases, raaga facts,
curriculum progress.  ``ResearchAgent`` fills it, and the composer reads
it - ``learned_phrase_bank`` is what turns a heard recording into material
for a tune.  ``knowledge_base.db`` is the durable Knowledge Base, the store
the specification calls the permanent learned memory, where a claim carries
its evidence, a second source agreeing attaches to the first rather than
duplicating it, and a source disagreeing is recorded as a conflict rather
than overwriting anything.

The Training queue already bridged into it (``training/knowledge_base.py``).
The listening path did not, so everything the agent learned by ear stopped
at its own repository: 222 items in the Knowledge Base, none of them from
audio.  This is the other half of that bridge.

What crosses, and how it is labelled:

``prayoga``     a phrase the agent heard, as a PATTERN, with the recording
                as evidence and ``audio_derived`` as the method - the KB's
                own way of saying "observed, not reasoned"
``observed_*``  what the ascent, descent and resting notes actually sounded
                like, as FACT

Nothing here invents a claim.  The confidence that reaches the Knowledge
Base is the confidence the analysis gave, and the evidence names the file
and the second it was heard at, so a wrong phrase can be traced back to the
recording that taught it.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..core.logging_setup import get_logger

log = get_logger("agent.kb_bridge")

#: Facts from ``_observe_facts`` worth making permanent.  These say what a
#: recording *sounded like*, which is a different claim from what the
#: library asserts, and the difference is the point of listening.
OBSERVED_KEYS = ("observed_ascent", "observed_descent",
                 "observed_resting_notes", "observed_tempo")


def publish(kb, source, raaga: str, phrases: Sequence[Any],
            facts: Sequence[Any] = (), run_id: str = "") -> int:
    """Commit what one recording taught to the Knowledge Base.

    ``kb`` may be ``None`` - an agent built without one still learns into
    its own repository, and this simply does nothing.  Nothing raised here
    is allowed to fail an ingestion: the recording has already been heard
    and stored, and losing the durable copy is worth a warning, not the
    loss of the work.
    """
    if kb is None or not (phrases or facts):
        return 0
    try:
        from ..kb.models import (Evidence, ExtractionMethod, KnowledgeItem,
                                 KnowledgeType, Scope)
        from ..kb.models import Source as KBSource
        from ..kb import normalize as kb_normalize
    except Exception as exc:  # noqa: BLE001
        log.warning("the Knowledge Base models could not be loaded: %s", exc)
        return 0

    try:
        kb_source = kb.add_source(KBSource(
            source_type="recording",
            title=getattr(source, "title", "") or "a recording",
            reference=getattr(source, "locator", ""),
            license_or_access_notes=getattr(source, "rights_status", ""),
            metadata={"provider": getattr(source, "provider", ""),
                      "heard": True}))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not register the recording with the Knowledge "
                    "Base: %s", exc)
        return 0

    quality = float(getattr(source, "quality", 0.6) or 0.6)
    written = 0
    for phrase in phrases:
        text = " ".join(phrase.swaras)
        written += _commit(
            kb, KnowledgeItem(
                canonical_name=raaga,
                knowledge_type=KnowledgeType.PATTERN,
                subject=raaga, predicate="prayoga",
                object_value=text, statement=text,
                structured_value=kb_normalize.structured_for("prayoga", text),
                scope=[Scope.CARNATIC, Scope.RAGA],
                raga=raaga, tags=["prayoga", "heard"],
                learned_by="listening"),
            Evidence(source_id=kb_source.source_id,
                     transcript_excerpt=text,
                     strength=float(getattr(phrase, "confidence", 0.5)),
                     extraction_method=ExtractionMethod.AUDIO,
                     run_id=run_id),
            quality, run_id)

    for fact in facts:
        key = getattr(fact, "key", "")
        if key not in OBSERVED_KEYS:
            continue
        value = str(getattr(fact, "value", ""))
        written += _commit(
            kb, KnowledgeItem(
                canonical_name=raaga,
                knowledge_type=KnowledgeType.FACT,
                subject=raaga, predicate=key,
                object_value=value, statement=f"{key.replace('_', ' ')}: {value}",
                structured_value=kb_normalize.structured_for(key, value),
                scope=[Scope.CARNATIC, Scope.RAGA],
                raga=raaga, tags=[key, "heard"],
                learned_by="listening"),
            Evidence(source_id=kb_source.source_id,
                     transcript_excerpt=value,
                     strength=float(getattr(fact, "confidence", 0.5)),
                     extraction_method=ExtractionMethod.AUDIO,
                     run_id=run_id),
            quality, run_id)

    if written:
        log.info("Knowledge Base: %d item(s) from listening to %s",
                 written, getattr(source, "title", "a recording"))
    return written


def _commit(kb, item, evidence, quality: float, run_id: str) -> int:
    """One item, with everything that can go wrong kept local to it."""
    try:
        result = kb.commit_knowledge(item, [evidence], source_quality=quality,
                                     run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - one bad item must not stop the rest
        log.warning("could not commit %s/%s to the Knowledge Base: %s",
                    item.subject, item.predicate, exc)
        return 0
    return 1 if getattr(result, "stored", False) else 0
