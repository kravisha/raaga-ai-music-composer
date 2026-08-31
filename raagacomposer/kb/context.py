"""Assembling what a task actually needs - specification section 19.

The Knowledge Base will grow, and a naive "give the model everything about
Kambhoji" gets slower and worse at the same time.  This builds the *smallest
sufficient* set: what the task needs, ranked, deduplicated, with the
constraints and the cautions that go with it, and with provenance where it
matters.

The judgement in here is what "sufficient" means, and it is task-shaped rather
than size-shaped.  Composing in a raga needs its structure and its phrases and
the things it must not do; it does not need the teaching notes about how to
introduce it to a beginner.  Explaining a choice needs provenance on every
item; generating a melody does not.  So a task profile says which categories
matter, how many of each are worth carrying, and whether provenance travels
with them.

Two things are always carried whatever the profile says, because leaving them
out is how a Knowledge Base becomes confidently wrong:

*Constraints.*  A rule about what the raga must not do is worth more than
another example of what it does.
*Disagreements.*  Where the sources conflict, the caller is told, rather than
being handed the more confident of two answers as though it were settled.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from .models import (ConflictState, KnowledgeItem, KnowledgeType, Status,
                     new_id)
from .retrieval import HybridRetriever, Hit, Query
from .service import KnowledgeBaseService

log = get_logger("kb.context")


@dataclass
class TaskProfile:
    """What one kind of work needs from the Knowledge Base."""

    name: str
    categories: Sequence[str] = ()
    per_category: int = 4
    total: int = 24
    include_provenance: bool = False
    include_examples: bool = True
    graph_depth: int = 1
    min_confidence: float = 0.0


PROFILES: Dict[str, TaskProfile] = {
    "compose": TaskProfile(
        "compose",
        categories=("scale", "phrase", "grammar", "ornament", "tonic",
                    "tempo", "structure"),
        per_category=5, total=28, graph_depth=2, min_confidence=0.25),
    "melody": TaskProfile(
        "melody",
        categories=("scale", "phrase", "grammar", "ornament"),
        per_category=6, total=24, graph_depth=2, min_confidence=0.25),
    "sing": TaskProfile(
        "sing", categories=("ornament", "phrase", "tempo", "scale"),
        per_category=5, total=20, graph_depth=1),
    "teach": TaskProfile(
        "teach",
        categories=("scale", "phrase", "grammar", "practice", "mood",
                    "ornament"),
        per_category=4, total=24, include_provenance=True, graph_depth=2),
    "explain": TaskProfile(
        "explain", per_category=6, total=30, include_provenance=True,
        graph_depth=2),
    "evaluate": TaskProfile(
        "evaluate", categories=("scale", "grammar", "phrase", "ornament"),
        per_category=5, total=20, min_confidence=0.3),
}
DEFAULT_PROFILE = TaskProfile("general", per_category=4, total=20)


@dataclass
class KnowledgeContext:
    """What was assembled, and everything needed to explain it afterwards."""

    context_id: str = field(default_factory=lambda: new_id("ctx"))
    task: str = ""
    subject: str = ""
    items: List[KnowledgeItem] = field(default_factory=list)
    constraints: List[KnowledgeItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    built_at: float = field(default_factory=time.time)

    @property
    def knowledge_ids(self) -> List[str]:
        return [i.knowledge_id for i in self.items + self.constraints]

    @property
    def is_empty(self) -> bool:
        return not self.items and not self.constraints

    def render(self) -> str:
        """The context as text a model or a person can read."""
        lines: List[str] = []
        if self.subject:
            lines.append(f"What is known about {self.subject}, "
                         f"for {self.task}:")
        if self.is_empty:
            lines.append("  Nothing is held about this yet.")
        for item in self.items:
            confidence = f"[{item.band.lower()}]"
            lines.append(f"  - {item.display()} {confidence}")
        if self.constraints:
            lines.append("")
            lines.append("Must not:")
            for item in self.constraints:
                lines.append(f"  - {item.display()}")
        if self.disagreements:
            lines.append("")
            lines.append("Sources disagree about:")
            for text in self.disagreements:
                lines.append(f"  - {text}")
        if self.warnings:
            lines.append("")
            lines.append("Cautions:")
            for text in self.warnings:
                lines.append(f"  - {text}")
        if self.unknowns:
            lines.append("")
            lines.append("Not known:")
            for text in self.unknowns:
                lines.append(f"  - {text}")
        return "\n".join(lines)


class KnowledgeContextBuilder:
    """Section 19.  Task in, smallest sufficient knowledge out."""

    def __init__(self, service: KnowledgeBaseService,
                 retriever: Optional[HybridRetriever] = None) -> None:
        self.service = service
        self.retriever = retriever or HybridRetriever(service)

    # ==================================================================
    def build(self, task: str, *, subject: str = "", raga: str = "",
              tala: str = "", text: str = "",
              profile: Optional[TaskProfile] = None,
              record_usage: bool = True) -> KnowledgeContext:
        profile = profile or PROFILES.get(task, DEFAULT_PROFILE)
        subject = subject or raga or tala
        context = KnowledgeContext(task=task, subject=subject)

        hits = self.retriever.search(Query(
            text=text or subject, subject=subject, raga=raga, tala=tala,
            graph_depth=profile.graph_depth,
            min_confidence=profile.min_confidence,
            limit=max(profile.total * 3, 60)))

        if not hits:
            context.unknowns.append(
                f"nothing is held about {subject or text or 'this'} yet")
            self._note_gaps(context, subject)
            return context

        # -- constraints first: they are carried whatever the profile says
        constraints = [h for h in hits
                       if h.item.knowledge_type == KnowledgeType.CONSTRAINT
                       or h.item.predicate in ("avoid", "forbidden",
                                               "must_not")]
        context.constraints = self._dedupe(
            [h.item for h in constraints])[:profile.per_category * 2]
        chosen_ids = {i.knowledge_id for i in context.constraints}

        # -- then the categories this task actually uses -----------------
        remaining = [h for h in hits if h.item.knowledge_id not in chosen_ids]
        if profile.categories:
            per_category: Dict[str, List[Hit]] = {}
            for hit in remaining:
                bucket = hit.item.predicate or hit.item.knowledge_type
                category = self._category_for(hit.item, profile.categories)
                if category is None:
                    continue
                per_category.setdefault(category, []).append(hit)
            picked: List[Hit] = []
            for category in profile.categories:
                picked.extend(per_category.get(category, [])[:profile.per_category])
            # Anything strongly scored that no category claimed still counts.
            claimed = {h.item.knowledge_id for h in picked}
            picked.extend([h for h in remaining
                           if h.item.knowledge_id not in claimed
                           and h.score >= 1.0])
        else:
            picked = remaining

        if not profile.include_examples:
            picked = [h for h in picked
                      if h.item.knowledge_type != KnowledgeType.EXAMPLE]

        picked.sort(key=lambda h: (-h.score, -h.item.importance))
        context.items = self._dedupe([h.item for h in picked])[:profile.total]

        # -- what the caller must be told about ---------------------------
        self._note_disagreements(context)
        self._note_warnings(context)
        self._note_gaps(context, subject)

        if profile.include_provenance:
            context.provenance = {
                item.knowledge_id: self.service.provenance(item.knowledge_id)
                for item in context.items[:12]}

        if record_usage:
            self.retriever.record_usage(task, text or subject,
                                        context.knowledge_ids,
                                        context.context_id)
        log.debug("context for %s/%s: %d item(s), %d constraint(s)", task,
                  subject, len(context.items), len(context.constraints))
        return context

    # ==================================================================
    @staticmethod
    def _category_for(item: KnowledgeItem,
                      categories: Sequence[str]) -> Optional[str]:
        """Which of the task's categories this item belongs to, if any."""
        predicate = (item.predicate or "").lower()
        mapping = {
            "arohanam": "scale", "avarohanam": "scale", "swaras": "scale",
            "prayoga": "phrase", "phrase": "phrase",
            "gamaka": "ornament", "jeeva": "structure", "nyasa": "structure",
            "rasa": "mood", "tempo": "tempo", "tonic": "tonic",
            "avoid": "grammar", "forbidden": "grammar",
            "varisai": "practice", "exercise": "practice",
        }
        category = mapping.get(predicate)
        if category is None:
            by_type = {
                KnowledgeType.PATTERN: "phrase",
                KnowledgeType.CONSTRAINT: "grammar",
                KnowledgeType.PROCEDURE: "practice",
                KnowledgeType.EXAMPLE: "phrase",
            }
            category = by_type.get(item.knowledge_type)
        return category if category in categories else None

    @staticmethod
    def _dedupe(items: Sequence[KnowledgeItem]) -> List[KnowledgeItem]:
        """Section 19 step 7.  The same claim twice helps nobody."""
        from . import normalize

        seen: set = set()
        out: List[KnowledgeItem] = []
        for item in items:
            key = normalize.normalise_statement(item.display())
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _note_disagreements(self, context: KnowledgeContext) -> None:
        """Never hand over the more confident of two answers as settled."""
        ids = set(context.knowledge_ids)
        for conflict in self.service.conflicts(open_only=True, limit=50):
            if conflict.claim_a not in ids and conflict.claim_b not in ids:
                continue
            first = self.service.get_by_id(conflict.claim_a)
            second = self.service.get_by_id(conflict.claim_b)
            if first is None or second is None:
                continue
            state = ("sources disagree"
                     if conflict.resolution_status == ConflictState.UNRESOLVED
                     else "true in different contexts")
            context.disagreements.append(
                f"{state}: \"{first.display()}\" ({first.confidence:.2f}) "
                f"against \"{second.display()}\" ({second.confidence:.2f}). "
                f"{conflict.notes}")

    def _note_warnings(self, context: KnowledgeContext) -> None:
        weak = [i for i in context.items if i.confidence < 0.4]
        if weak:
            context.warnings.append(
                f"{len(weak)} of these are weakly supported "
                f"(confidence below 0.4) - treat them as provisional")
        unheard = [i for i in context.items
                   if "stated" in i.tags and "heard" not in i.tags]
        if unheard:
            context.warnings.append(
                f"{len(unheard)} of these were stated in a lesson rather than "
                f"heard, so nothing has verified them by ear")
        disputed = [i for i in context.items if i.status == Status.DISPUTED]
        if disputed:
            context.warnings.append(
                f"{len(disputed)} of these are disputed and a person has not "
                f"yet decided")

    def _note_gaps(self, context: KnowledgeContext, subject: str) -> None:
        """Section 30 read the other way: say what is missing here."""
        if not subject:
            return
        for gap in self.service.gaps(open_only=True, limit=50):
            from . import normalize
            if normalize.normalise_name(gap.subject) == \
                    normalize.normalise_name(subject):
                text = gap.missing_information.strip()
                if gap.reason.strip():
                    text = f"{text} ({gap.reason.strip()})"
                context.unknowns.append(text)
