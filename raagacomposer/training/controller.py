"""The Training tab's controller - specification section 17.

Everything the tab can do is a method here, and none of it needs Qt.  That is
the same split the rest of the application uses: ``app.py`` has no Qt imports
and neither does this, so the whole feature can be driven and tested without a
window, and the panel is only a view.

It also owns the one decision the specification is most insistent about, in
section 11: autonomous search does not mean autonomous approval.  This class
will happily suggest a search phrase from a curriculum gap, but nothing enters
the queue except through :meth:`add_to_queue`, which takes the sources a
person picked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from ..raaga.library import RaagaLibrary, library as default_library
from .knowledge_base import KnowledgeBaseService
from .models import (Accessibility, KnowledgeEntry, LearningReport,
                     LearningRun, LearningSource, Objective, RunStatus,
                     SearchQuery)
from .objectives import LearningObjectiveService
from .pipeline import LearningPipeline
from .queue import TrainingQueueService
from .report import LearningReportService, TrainingHistoryService
from .search import LearningSourceSearchService, match_raaga
from .store import TrainingStore

log = get_logger("training.controller")


class TrainingController:
    """One object the Training tab talks to."""

    def __init__(self, settings: Optional[Settings] = None,
                 raagas: Optional[RaagaLibrary] = None,
                 store: Optional[TrainingStore] = None,
                 agent_repo=None, curriculum=None,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        self.settings = settings or Settings.load()
        self.raagas = raagas or default_library()
        path = getattr(self.settings, "training_db", "") or None
        self.store = store or TrainingStore(Path(path) if path else None)
        self.agent_repo = agent_repo
        self.curriculum = curriculum

        self.search_service = LearningSourceSearchService(
            self.store, self.raagas, self.settings)
        self.objective_service = LearningObjectiveService(
            self.store, self.raagas, agent_repo, curriculum)
        self.pipeline = LearningPipeline(self.store, self.raagas,
                                         self.settings, agent_repo, curriculum)
        self.queue = TrainingQueueService(self.store, self.pipeline, on_change)
        self.reports = LearningReportService(self.store)
        self.history = TrainingHistoryService(self.store)
        self.knowledge = KnowledgeBaseService(self.store, self.raagas,
                                              agent_repo)

        self.last_results: List[LearningSource] = []
        self.last_query: Optional[SearchQuery] = None

    # ==================================================================
    # search (section 3.1, 3.2)
    # ==================================================================
    def search(self, phrase: str, **filters) -> List[LearningSource]:
        query = SearchQuery(phrase=phrase, **{
            k: v for k, v in filters.items()
            if k in SearchQuery.__dataclass_fields__})
        self.last_query = query
        self.last_results = self.search_service.search(query)
        return self.last_results

    def clear_search(self) -> None:
        self.last_results = []
        self.last_query = None

    def search_history(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self.store.searches(limit)

    def suggested_phrase(self) -> str:
        """Section 11 - a phrase drawn from where study has actually got to.

        This *suggests*; it never searches or approves on its own.
        """
        raaga = ""
        if self.curriculum is not None:
            try:
                raaga = self.curriculum.current_raaga() or ""
            except Exception:  # noqa: BLE001
                raaga = ""
        if not raaga:
            raaga = getattr(self.settings, "pilot_raaga", "") or ""
        if not raaga:
            return "Carnatic music beginner lessons"

        missing = self._missing_facts(raaga)
        if missing:
            return f"{raaga} {missing[0]} lesson"
        return f"{raaga} characteristic phrases"

    def _missing_facts(self, raaga: str) -> List[str]:
        if self.agent_repo is None:
            return []
        try:
            known = {f.key for f in self.agent_repo.facts(raaga)}
        except Exception:  # noqa: BLE001
            return []
        return [key for key in ("arohanam", "avarohanam", "gamaka", "jeeva")
                if key not in known]

    # ==================================================================
    # approval and the queue (section 3.3, 20 rule 1)
    # ==================================================================
    def add_to_queue(self, source_ids: Sequence[str]) -> List[LearningRun]:
        """Only what a person chose. Nothing else ever reaches the queue."""
        chosen: List[LearningSource] = []
        for source_id in source_ids:
            source = self.store.candidate(source_id)
            if source is not None:
                chosen.append(source)
        if not chosen:
            return []
        phrase = self.last_query.phrase if self.last_query else ""
        runs = self.queue.enqueue(
            chosen, phrase,
            objectives_for=lambda s: self.objective_service.suggest(s, phrase))
        log.info("%d source(s) approved for learning", len(runs))
        return runs

    def already_learned(self, source_id: str) -> Optional[LearningRun]:
        source = self.store.candidate(source_id)
        if source is None:
            return None
        return self.store.completed_run_for(source)

    def relearn(self, source_id: str) -> Optional[LearningRun]:
        source = self.store.candidate(source_id)
        if source is None:
            return None
        previous = self.store.completed_run_for(source)
        phrase = self.last_query.phrase if self.last_query else ""
        run = self.queue.relearn(source, phrase,
                                 previous.run_id if previous else "")
        self.objective_service.set_objectives(
            run.run_id, self.objective_service.suggest(source, phrase))
        return run

    # -- queue controls ------------------------------------------------
    def start_learning(self) -> None:
        self.queue.start()

    def pause_learning(self) -> None:
        self.queue.pause()

    def resume_learning(self) -> None:
        self.queue.resume()

    def cancel_current(self) -> None:
        self.queue.cancel_current()

    def remove_from_queue(self, run_id: str) -> None:
        self.queue.remove(run_id)

    def retry(self, run_id: str) -> None:
        self.queue.retry(run_id)

    def skip(self, run_id: str) -> None:
        self.queue.skip(run_id)

    def queue_snapshot(self) -> List[Dict[str, Any]]:
        return self.queue.snapshot()

    def learn_one_now(self) -> Optional[LearningReport]:
        """Work the next source on this thread - used by tests and the CLI."""
        return self.queue.process_next()

    # ==================================================================
    # objectives (section 6)
    # ==================================================================
    def objectives(self, run_id: str) -> List[Objective]:
        return self.objective_service.objectives_for_run(run_id)

    def set_objectives(self, run_id: str,
                       objectives: Sequence[Objective]) -> List[Objective]:
        return self.objective_service.set_objectives(run_id, objectives)

    def add_objective(self, run_id: str, description: str,
                      category: str = "general") -> List[Objective]:
        return self.objective_service.add_objective(run_id, description,
                                                    category)

    def remove_objective(self, run_id: str, objective_id: str
                         ) -> List[Objective]:
        return self.objective_service.remove_objective(run_id, objective_id)

    # ==================================================================
    # supplying what a source could not give us (section 4)
    # ==================================================================
    def supply_file(self, source_id: str, path: str) -> bool:
        """The honest way to make an unreachable source learnable."""
        source = self.store.candidate(source_id)
        if source is None:
            return False
        file_path = Path(path)
        if not file_path.exists():
            return False
        self.store.update_candidate(
            source_id, local_path=str(file_path),
            accessibility_status=Accessibility.ACCESSIBLE)
        self.store.audit("source.file_supplied", str(file_path),
                         source_id=source_id)
        return True

    def supply_transcript(self, source_id: str, text: str) -> bool:
        source = self.store.candidate(source_id)
        if source is None or not text.strip():
            return False
        metadata = dict(source.metadata, transcript=text.strip())
        self.store.update_candidate(
            source_id, metadata=metadata,
            accessibility_status=Accessibility.TRANSCRIPT,
            transcript_available=True)
        self.store.audit("source.transcript_supplied",
                         f"{len(text.split())} word(s)", source_id=source_id)
        return True

    # ==================================================================
    # reports, history and the knowledge base
    # ==================================================================
    def report(self, run_id: str) -> Optional[LearningReport]:
        return self.reports.report(run_id)

    def render_report(self, run_id: str) -> str:
        return self.reports.render_run(run_id)

    def training_history(self, **filters) -> List[Dict[str, Any]]:
        return self.history.entries(**filters)

    def totals(self) -> Dict[str, Any]:
        return self.history.totals()

    def search_knowledge(self, **criteria) -> List[KnowledgeEntry]:
        return self.knowledge.search(**criteria)

    def provenance(self, knowledge_id: str) -> Dict[str, Any]:
        return self.knowledge.provenance(knowledge_id)

    def mark_knowledge_incorrect(self, knowledge_id: str,
                                 reason: str = "") -> None:
        self.knowledge.mark_incorrect(knowledge_id, reason)

    def approve_knowledge(self, knowledge_id: str, note: str = "") -> None:
        self.knowledge.approve(knowledge_id, note)

    def conflicts(self, unresolved_only: bool = True):
        return self.store.conflicts(unresolved_only=unresolved_only)

    def resolve_conflict(self, conflict_id: str, resolution: str) -> None:
        self.store.resolve_conflict(conflict_id, resolution)

    def delete_run(self, run_id: str) -> None:
        self.queue.remove(run_id)

    # ==================================================================
    def close(self) -> None:
        self.queue.stop()
        if not self.store.closed:
            self.store.close()
