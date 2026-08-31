"""The whole of section 7, in order, for one source.

Phases A to G run here.  Each one reports its status before it starts, so a
creator watching the queue sees where a source actually is rather than a
spinner; each one can fail without losing the work the earlier ones did.

Two rules from section 20 shape the flow more than anything else:

rule 9   inaccessible content is reported honestly.  A source we could only
         read the description of does not quietly produce a thin report - it
         produces a report that says, in those words, that the content was not
         analysed, and it offers the two ways the creator can fix that.
rule 4   every completed source has a report.  That includes the ones that
         failed: a source that taught us nothing still has to say so, and why.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..raaga.library import RaagaLibrary
from .access import AccessDecision, SourceAccessService
from .acquisition import MediaIngestionService
from .knowledge_base import KnowledgeBaseService
from .models import (Accessibility, LearningReport, LearningRun,
                     LearningSource, Objective, ObjectiveStatus, RunStatus)
from .objectives import LearningObjectiveService
from .semantics import Interpretation, Observation, SemanticLearningService
from .store import TrainingStore
from .validation import KnowledgeValidationService

log = get_logger("training.pipeline")

#: What the creator is told when only the description was available.  The
#: specification asks for this wording; it is a constant so it cannot drift.
METADATA_ONLY_NOTICE = "METADATA ONLY - CONTENT NOT ANALYZED"

ProgressHook = Callable[[str, float, str], None]


class LearningPipeline:
    """Phases A to G for one source, with a report at the end whatever happens."""

    def __init__(self, store: TrainingStore, raagas: RaagaLibrary,
                 settings=None, agent_repo=None, curriculum=None,
                 kb=None) -> None:
        self.store = store
        self.raagas = raagas
        self.settings = settings
        self.access = SourceAccessService(store)
        self.media = MediaIngestionService(raagas, settings)
        self.semantics = SemanticLearningService(raagas)
        self.validation = KnowledgeValidationService(store)
        self.knowledge = KnowledgeBaseService(store, raagas, agent_repo,
                                             kb=kb)
        self.objectives = LearningObjectiveService(store, raagas, agent_repo,
                                                   curriculum)

    # ==================================================================
    def run(self, run: LearningRun, source: LearningSource,
            on_progress: Optional[ProgressHook] = None,
            cancelled: Optional[Callable[[], bool]] = None) -> LearningReport:
        def step(status: str, progress: float, detail: str) -> None:
            self.store.update_run(run.run_id, status=status,
                                  progress=progress, detail=detail)
            if on_progress:
                on_progress(status, progress, detail)

        def stopped() -> bool:
            return bool(cancelled and cancelled())

        self.store.update_run(run.run_id, started_at=time.time(),
                              attempts=run.attempts + 1)

        # -- phase A ---------------------------------------------------
        step(RunStatus.CHECKING_ACCESS, 0.05, "checking what can be reached")
        decision = self.access.check(source)
        self.access.record_provenance(source, run.run_id, decision)

        objectives = self.store.objectives(run.run_id)
        if not objectives:
            objectives = self.objectives.set_objectives(
                run.run_id, self.objectives.suggest(source, run.search_phrase))

        if stopped():
            return self._abandon(run, source, objectives, "cancelled")

        if not decision.analysable:
            return self._not_analysed(run, source, objectives, decision)

        # -- phase B ---------------------------------------------------
        step(RunStatus.FETCHING_METADATA, 0.15, decision.reason)
        step(RunStatus.ACQUIRING_TRANSCRIPT if
             decision.representation == "transcript" else
             RunStatus.PREPARING_CONTENT, 0.3, "acquiring the content")
        max_seconds = float(getattr(self.settings,
                                    "learning_max_audio_seconds", 120.0)
                            if self.settings else 120.0)
        content = self.media.acquire(source, decision, max_seconds)
        if not content.ok:
            return self._failed(run, source, objectives,
                                content.error or "the content could not be read")

        if stopped():
            return self._abandon(run, source, objectives, "cancelled")

        # -- phases C, D and E ------------------------------------------
        step(RunStatus.ANALYZING, 0.5, content.describe())
        interpretation = self.semantics.interpret(source, content, objectives)

        step(RunStatus.EXTRACTING_KNOWLEDGE, 0.7,
             f"{len(interpretation.observations)} observation(s)")
        if stopped():
            return self._abandon(run, source, objectives, "cancelled")

        # -- phase F ----------------------------------------------------
        step(RunStatus.VALIDATING, 0.8, "checking against what is already held")
        objective_ids = {o.category: o.objective_id for o in objectives}
        quality = 0.85 if content.representation == "exercise" else 0.7
        outcome = self.validation.validate(
            interpretation.observations, source, run.run_id, objective_ids,
            source_quality=quality)

        # -- phase G ----------------------------------------------------
        step(RunStatus.SAVING_KNOWLEDGE, 0.9,
             f"{len(outcome.accepted)} item(s) to store")
        knowledge_ids = self.knowledge.store_all(outcome.accepted, source,
                                                 run.run_id)
        for conflict in outcome.conflicts:
            self.store.add_conflict(conflict)

        self._mark_objectives(objectives, interpretation, outcome)
        self.store.save_objectives(run.run_id, objectives)

        report = self._report(run, source, objectives, interpretation, outcome,
                              knowledge_ids, content.describe())
        self.store.save_report(report)
        result = (f"{len(knowledge_ids)} item(s) learned, "
                  f"{len(outcome.confirmed)} confirmed, "
                  f"{len(outcome.conflicts)} conflict(s)")
        self.store.update_run(run.run_id, status=RunStatus.COMPLETED,
                              progress=1.0, detail=result, result=result,
                              completed_at=time.time())
        step(RunStatus.COMPLETED, 1.0, result)
        return report

    # ==================================================================
    # outcomes that are not a clean completion
    # ==================================================================
    def _not_analysed(self, run: LearningRun, source: LearningSource,
                      objectives: List[Objective],
                      decision: AccessDecision) -> LearningReport:
        """Section 4: say plainly that the lesson was not learned."""
        for objective in objectives:
            objective.status = ObjectiveStatus.NOT_STARTED
            objective.outcome = "the content was never analysed"

        report = LearningReport(
            run_id=run.run_id, source=source, objectives=objectives,
            summary=f"{METADATA_ONLY_NOTICE}. {decision.reason}.",
            understood=(
                f"Only what this source says about itself was available: "
                f"{source.description or 'no description was given'} "
                f"Nothing was fetched and nothing was heard, so the lesson "
                f"itself has not been learned."),
            learned=[], confirmed=[], confidence=0.0,
            analysed_representation="none",
            honest_limits=[decision.reason],
            next_learning=[f"{offer} to let this source be analysed"
                           for offer in decision.offers])
        self.store.save_objectives(run.run_id, objectives)
        self.store.save_report(report)

        status = (RunStatus.SOURCE_INACCESSIBLE
                  if decision.status in (Accessibility.NOT_ACCESSIBLE,
                                         Accessibility.UNSUPPORTED,
                                         Accessibility.USER_FILE_REQUIRED)
                  else RunStatus.COMPLETED)
        self.store.update_run(run.run_id, status=status, progress=1.0,
                              detail=METADATA_ONLY_NOTICE,
                              result=METADATA_ONLY_NOTICE,
                              completed_at=time.time())
        self.store.audit("run.not_analysed", decision.reason,
                         run_id=run.run_id, source_id=source.source_id)
        return report

    def _failed(self, run: LearningRun, source: LearningSource,
                objectives: List[Objective], error: str) -> LearningReport:
        """Section 14: preserve partial work, allow a retry."""
        report = LearningReport(
            run_id=run.run_id, source=source, objectives=objectives,
            summary=f"This source could not be processed: {error}",
            understood="Nothing was analysed, so nothing was understood.",
            confidence=0.0, analysed_representation="none",
            honest_limits=[error],
            next_learning=["Retry, or supply the file yourself"])
        self.store.save_report(report)
        self.store.update_run(run.run_id, status=RunStatus.FAILED,
                              progress=1.0, detail=error, error=error,
                              result=f"failed: {error}",
                              completed_at=time.time())
        self.store.audit("run.failed", error, run_id=run.run_id,
                         source_id=source.source_id)
        return report

    def _abandon(self, run: LearningRun, source: LearningSource,
                 objectives: List[Objective], reason: str) -> LearningReport:
        report = LearningReport(
            run_id=run.run_id, source=source, objectives=objectives,
            summary=f"Stopped before finishing: {reason}.",
            understood="This source was not carried through to a conclusion.",
            confidence=0.0, analysed_representation="none",
            honest_limits=[reason],
            next_learning=["Retry this source when you are ready"])
        self.store.save_report(report)
        self.store.update_run(run.run_id, status=RunStatus.SKIPPED,
                              progress=0.0, detail=reason, result=reason,
                              completed_at=time.time())
        return report

    # ==================================================================
    def _mark_objectives(self, objectives: Sequence[Objective],
                         interpretation: Interpretation, outcome) -> None:
        """Each objective gets a verdict, including 'not in this source'."""
        by_category: Dict[str, List[Observation]] = {}
        for observation in interpretation.observations:
            by_category.setdefault(observation.objective_category,
                                   []).append(observation)

        rejected = {id(o) for o, _ in outcome.rejected}
        uncertain = {id(o) for o in outcome.uncertain}

        for objective in objectives:
            found = by_category.get(objective.category, [])
            usable = [o for o in found
                      if id(o) not in rejected and id(o) not in uncertain]
            if usable:
                confidence = sum(o.confidence for o in usable) / len(usable)
                objective.status = (ObjectiveStatus.LEARNED if confidence >= 0.5
                                    else ObjectiveStatus.PARTIAL)
                objective.confidence = round(confidence, 3)
                objective.evidence = "; ".join(
                    o.evidence for o in usable[:3] if o.evidence)
                objective.outcome = " ".join(o.statement for o in usable[:3])
            elif found:
                objective.status = ObjectiveStatus.UNCERTAIN
                objective.confidence = round(
                    max(o.confidence for o in found), 3)
                objective.evidence = found[0].evidence
                objective.outcome = (
                    "something bearing on this was noticed, but not clearly "
                    "enough to be written down as knowledge")
            else:
                objective.status = ObjectiveStatus.NOT_PRESENT
                objective.confidence = 0.0
                objective.outcome = "this source did not cover it"

    # ------------------------------------------------------------------
    def _report(self, run: LearningRun, source: LearningSource,
                objectives: Sequence[Objective],
                interpretation: Interpretation, outcome,
                knowledge_ids: Sequence[str],
                representation: str) -> LearningReport:
        confidences = [e.confidence for e in outcome.accepted]
        confidence = round(sum(confidences) / len(confidences), 3) \
            if confidences else 0.0

        limits = list(interpretation.limits)
        for observation in outcome.uncertain:
            limits.append(f"not certain enough to record: "
                          f"{observation.statement}")
        for observation, reason in outcome.rejected:
            limits.append(f"set aside - {reason}")

        return LearningReport(
            run_id=run.run_id, source=source, objectives=list(objectives),
            summary=interpretation.summary,
            understood=interpretation.understood,
            learned=[e.normalized_statement for e in outcome.accepted],
            confirmed=[e.normalized_statement for e in outcome.confirmed],
            conflicts=list(outcome.conflicts),
            practical_application=self._application(outcome, source),
            confidence=confidence,
            next_learning=self._next_learning(objectives, outcome, source),
            knowledge_ids=list(knowledge_ids),
            analysed_representation=representation,
            honest_limits=limits)

    # ------------------------------------------------------------------
    @staticmethod
    def _application(outcome, source: LearningSource) -> List[str]:
        """Section 8.8 - what this actually changes about the music."""
        out: List[str] = []
        categories = {e.category for e in outcome.accepted}
        heard_phrases = [e for e in outcome.accepted
                         if e.category == "phrase" and "heard" in e.tags]
        if heard_phrases:
            out.append(
                f"{len(heard_phrases)} phrase(s) were heard rather than read, "
                f"so they are available to the composer: melody generation "
                f"will quote them, and the evaluator will count them towards "
                f"phrase authenticity.")
        stated = [e for e in outcome.accepted
                  if e.category == "phrase" and "stated" in e.tags]
        if stated:
            out.append(
                f"{len(stated)} phrase(s) were stated rather than heard. They "
                f"are recorded here but are deliberately not given to the "
                f"composer, because nothing has verified them by ear.")
        if "tonic" in categories:
            out.append("The tonic found here informs how this material is "
                       "read; it does not change how new tunes are pitched.")
        if "tempo" in categories:
            out.append("The measured tempo can inform the pacing chosen for "
                       "this raaga.")
        if "ornament" in categories:
            out.append("Ornament guidance affects gamaka rendering when this "
                       "raaga is sung or played.")
        if "tala" in categories:
            out.append("The tala noted here bears on rhythmic arrangement.")
        if not out:
            out.append("Nothing here changes what the application plays yet.")
        return out

    @staticmethod
    def _next_learning(objectives: Sequence[Objective], outcome,
                       source: LearningSource) -> List[str]:
        """Section 8.10 - honest suggestions, drawn from what was missed."""
        out: List[str] = []
        missed = [o for o in objectives
                  if o.status in (ObjectiveStatus.NOT_PRESENT,
                                  ObjectiveStatus.UNCERTAIN)]
        for objective in missed[:3]:
            out.append(f"Still open: {objective.description} Look for a "
                       f"source that covers it.")
        if outcome.conflicts:
            out.append(f"{len(outcome.conflicts)} disagreement(s) with what "
                       f"was already held need a person to decide.")
        raga = str(source.metadata.get("raaga", ""))
        if raga and any(e.category == "phrase" for e in outcome.accepted):
            out.append(f"More phrase material for {raga} would raise phrase "
                       f"authenticity further; that is the weakest score the "
                       f"evaluator reports.")
        if not out:
            out.append("Nothing outstanding from this source.")
        return out
