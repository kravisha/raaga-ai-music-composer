"""What we mean to get out of a source - specification section 6.

Every source must have explicit objectives before it is learned from.  That is
not bookkeeping: it is what turns "the system watched a lesson" into a claim
that can be checked.  An objective names something to look for, the analysis
either finds it or does not, and the report says which - so a source that
taught us nothing is visibly a source that taught us nothing, rather than a
paragraph of prose that sounds like progress.

Objectives are suggested from four places, in the order the specification
lists them: what the creator searched for, what the source says about itself,
where the curriculum has got to, and what the knowledge base is still missing.
The creator may then edit, add or remove any of them before learning starts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..raaga.library import Raaga, RaagaLibrary
from .models import LearningSource, Objective, ObjectiveStatus
from .search import match_raaga, tokenize
from .store import TrainingStore

log = get_logger("training.objectives")

#: The catalogue of things worth looking for, each with the words that suggest
#: it and the knowledge category it would produce.  Priority 1 is highest.
TEMPLATES: Sequence[Dict[str, Any]] = (
    {"key": "raaga_identity", "priority": 1, "category": "raaga",
     "description": "Identify the raaga being taught.",
     "words": ("raaga", "raga", "ragam", "identify", "which")},
    {"key": "arohanam", "priority": 1, "category": "scale",
     "description": "Learn the arohanam and avarohanam.",
     "words": ("arohanam", "avarohanam", "scale", "ascent", "descent",
               "beginner", "basic", "introduction")},
    {"key": "prayoga", "priority": 1, "category": "phrase",
     "description": "Identify the characteristic prayogas.",
     "words": ("prayoga", "phrase", "sanchara", "characteristic", "idiom",
               "identity")},
    {"key": "gamaka", "priority": 2, "category": "ornament",
     "description": "Understand the important gamakas.",
     "words": ("gamaka", "ornament", "kampita", "oscillation", "technique")},
    {"key": "jeeva", "priority": 2, "category": "structure",
     "description": "Learn which swaras carry the raaga and where phrases "
                    "come to rest.",
     "words": ("jeeva", "nyasa", "resting", "cadence", "resolve", "important")},
    {"key": "tala", "priority": 2, "category": "tala",
     "description": "Identify the tala.",
     "words": ("tala", "talam", "adi", "rupaka", "misra", "beat", "rhythm",
               "cycle")},
    {"key": "tempo", "priority": 3, "category": "tempo",
     "description": "Learn the tempo the raaga is usually taken at.",
     "words": ("tempo", "speed", "kala", "pace", "laya")},
    {"key": "mood", "priority": 3, "category": "mood",
     "description": "Understand the mood the raaga carries.",
     "words": ("mood", "feel", "rasa", "emotion", "bhava", "usage")},
    {"key": "structure", "priority": 3, "category": "composition",
     "description": "Understand the composition structure.",
     "words": ("composition", "structure", "sangati", "sangathi", "kriti",
               "varnam", "pallavi", "charanam", "form")},
    {"key": "exercise", "priority": 3, "category": "practice",
     "description": "Learn the voice exercises being taught.",
     "words": ("varisai", "sarali", "janta", "exercise", "drill", "practice",
               "alankaram")},
    {"key": "avoid", "priority": 2, "category": "grammar",
     "description": "Learn the common mistakes and phrases to avoid.",
     "words": ("avoid", "mistake", "wrong", "common", "forbidden", "varjya")},
    {"key": "improvisation", "priority": 3, "category": "improvisation",
     "description": "Learn the improvisation technique demonstrated.",
     "words": ("alapana", "kalpana", "improvisation", "manodharma", "neraval",
               "swaram")},
)

#: The facts a raaga needs before the agent can be said to know it.  A gap
#: here becomes a high-priority objective, which is how a knowledge gap in
#: section 11 turns into something a source is actually asked to supply.
CORE_FACTS = ("arohanam", "avarohanam", "jeeva", "nyasa", "gamaka")

_FACT_OBJECTIVE = {
    "arohanam": "arohanam", "avarohanam": "arohanam", "jeeva": "jeeva",
    "nyasa": "jeeva", "gamaka": "gamaka",
}


class LearningObjectiveService:
    """Suggests objectives, and keeps them editable until learning starts."""

    def __init__(self, store: TrainingStore, raagas: RaagaLibrary,
                 knowledge_repo=None, curriculum=None) -> None:
        self.store = store
        self.raagas = raagas
        #: The agent's own musical memory, consulted for what is already known.
        self.knowledge_repo = knowledge_repo
        #: The agent's curriculum, consulted for where study has got to.
        self.curriculum = curriculum

    # ------------------------------------------------------------------
    def suggest(self, source: LearningSource, search_phrase: str = "",
                limit: int = 6) -> List[Objective]:
        """Objectives worth setting for this source, best first."""
        raaga = self._raaga_for(source, search_phrase)
        words = set(tokenize(f"{search_phrase} {source.title} "
                             f"{source.description}"))
        topic = str(source.metadata.get("topic", ""))

        scored: List[tuple] = []
        for template in TEMPLATES:
            score = 0.0
            hits = words & set(template["words"])
            if hits:
                score += 0.5 + 0.1 * min(3, len(hits))
            if topic and topic in template["key"]:
                score += 0.6
            if template["key"].startswith(topic or "\0"):
                score += 0.3
            score += self._gap_bonus(template["key"], raaga)
            # Priority breaks ties, so a first-principles objective is not
            # displaced by an incidental keyword match.
            score += (4 - template["priority"]) * 0.05
            if score > 0:
                scored.append((score, template))

        if not scored:
            # Nothing in the text told us anything.  Fall back to what any
            # lesson about a raaga should be asked for.
            scored = [(1.0, t) for t in TEMPLATES
                      if t["key"] in ("raaga_identity", "arohanam", "prayoga")]

        scored.sort(key=lambda pair: (-pair[0], pair[1]["priority"]))
        objectives: List[Objective] = []
        for score, template in scored[:limit]:
            description = template["description"]
            if raaga is not None and template["key"] != "raaga_identity":
                description = description.rstrip(".") + f" for {raaga.name}."
            objectives.append(Objective(
                description=description,
                category=template["category"],
                priority=int(template["priority"]),
                status=ObjectiveStatus.NOT_STARTED))
        log.debug("suggested %d objective(s) for %s", len(objectives),
                  source.title)
        return objectives

    # ------------------------------------------------------------------
    def _raaga_for(self, source: LearningSource,
                   search_phrase: str) -> Optional[Raaga]:
        named = str(source.metadata.get("raaga", ""))
        if named:
            found = self.raagas.get(named)
            if found is not None:
                return found
        return match_raaga(f"{search_phrase} {source.title}", self.raagas)

    def _gap_bonus(self, key: str, raaga: Optional[Raaga]) -> float:
        """Ask harder for what the agent is missing - section 11.

        A knowledge gap is the best possible reason to set an objective, so a
        fact the agent does not yet hold outranks one it merely matched a word
        against.
        """
        if raaga is None or self.knowledge_repo is None:
            return 0.0
        try:
            known = {f.key for f in self.knowledge_repo.facts(raaga.name)}
        except Exception:  # noqa: BLE001 - memory is advisory here
            return 0.0
        missing = {_FACT_OBJECTIVE[f] for f in CORE_FACTS if f not in known}
        return 0.45 if key in missing else 0.0

    # ------------------------------------------------------------------
    def objectives_for_run(self, run_id: str) -> List[Objective]:
        return self.store.objectives(run_id)

    def set_objectives(self, run_id: str,
                       objectives: Sequence[Objective]) -> List[Objective]:
        """Replace a run's objectives - the creator's edits land here."""
        kept = [o for o in objectives if o.description.strip()]
        self.store.save_objectives(run_id, kept)
        self.store.audit("objectives.set", f"{len(kept)} objective(s)",
                         run_id=run_id)
        return self.store.objectives(run_id)

    def add_objective(self, run_id: str, description: str,
                      category: str = "general", priority: int = 2
                      ) -> List[Objective]:
        current = self.store.objectives(run_id)
        current.append(Objective(description=description.strip(),
                                 category=category, priority=priority,
                                 user_defined=True))
        return self.set_objectives(run_id, current)

    def remove_objective(self, run_id: str,
                         objective_id: str) -> List[Objective]:
        current = [o for o in self.store.objectives(run_id)
                   if o.objective_id != objective_id]
        return self.set_objectives(run_id, current)

    def record_outcome(self, objective: Objective, status: str,
                       evidence: str = "", outcome: str = "",
                       confidence: float = 0.0) -> Objective:
        objective.status = status
        objective.evidence = evidence
        objective.outcome = outcome
        objective.confidence = round(float(confidence), 3)
        return objective
