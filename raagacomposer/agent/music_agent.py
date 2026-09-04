"""The music agent: the student behind the instrument.

Learning specification sections 3.1, 9, 15 and 16.  The agent owns the
knowledge repository, asks the curriculum what to study, sends the research
agent looking for permitted material, listens to it, practises, is marked by
the evaluator, and writes everything it learns back to permanent memory.

It also serves the composer: when the application asks for a raaga suggestion
or for a tune, the answer comes from what the agent has learned rather than
from a fixed table.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.models import CreativeBrief, Note
from ..core.settings import Settings
from ..raaga.library import Raaga, RaagaLibrary, library as default_library
from ..raaga.selection import expand_feel_words
from .curriculum import CurriculumEngine, Unit
from .evaluator import Evaluation, Evaluator, Finding
from .guidance import Guidance, build_guidance
from .knowledge import KnowledgeRepository, Lesson, Phrase
from .learned import (describe_knowledge, knowledge_confidence,
                      learned_phrase_bank, learned_raaga)
from .originality import PhraseIndex, check as check_originality
from .practice import PracticeEngine, PracticeReport, practice_seed
from .research import ResearchAgent, SourceCandidate

log = get_logger("agent.music")

NEGATIVE = re.compile(
    r"\b(not|isn't|doesn't|does not|no|never|wrong|bad|poor|boring|mechanical|"
    r"awful|worse|drift|off)\b", re.I)
POSITIVE = re.compile(
    r"\b(good|great|nice|lovely|like|love|keep|beautiful|amazing|better|"
    r"perfect|yes)\b", re.I)
# What a creator's complaint points at, in the evaluator's own vocabulary
# (agent/evaluator.py finding kinds), so guidance can act on it.
FEEDBACK_KINDS = [
    (re.compile(r"mechanical|robotic|monoton|boring|repetit|same"),
     ["repetitive", "no_gamaka"]),
    (re.compile(r"flat|lifeless|no (feel|ornament|gamaka)|plain"), ["no_gamaka"]),
    (re.compile(r"(not|n't|doesn't|does not) (sound|feel) like|drift|wrong raaga|"
                r"not (the )?raaga|sounds like (another|a different)"),
     ["neighbour_drift", "no_idiom"]),
    (re.compile(r"jump|leap|disjoint|jerky|all over"), ["too_many_leaps"]),
    (re.compile(r"(end|finish|clos)\w* (badly|abrupt|wrong|nowhere)|unresolved|"
                r"doesn't resolve|does not resolve"), ["no_cadence"]),
    (re.compile(r"off (the )?beat|rhythm|timing"), ["off_beat"]),
]


@dataclass
class LearningStep:
    at: float = field(default_factory=time.time)
    action: str = ""              # research | practice | idle | blocked
    unit_id: str = ""
    raaga: str = ""
    detail: str = ""
    score: float = 0.0
    passed: bool = False

    def summary(self) -> str:
        return f"{self.action}: {self.detail}"


@dataclass
class RaagaSuggestion:
    name: str
    score: float
    confidence: float
    reason: str
    evidence: List[str] = field(default_factory=list)
    learned: bool = False

    @property
    def rationale(self) -> str:
        """What the raaga panel shows: the reason plus how sure the agent is."""
        mark = "studied" if self.learned else "not studied yet"
        extra = f" [{'; '.join(self.evidence)}]" if self.evidence else ""
        return f"{self.reason} ({self.confidence:.0%} confident, {mark}){extra}"

    def describe(self) -> str:
        mark = "learned" if self.learned else "from the reference library"
        return f"{self.name} ({self.confidence:.0%} confident, {mark}) - {self.reason}"


class MusicAgent:
    """The persistent musician. One per application instance."""

    def __init__(self, settings: Optional[Settings] = None,
                 library: Optional[RaagaLibrary] = None,
                 repository: Optional[KnowledgeRepository] = None,
                 llm=None) -> None:
        self.settings = settings or Settings.load()
        self.library = library or default_library()
        configured_db = getattr(self.settings, "knowledge_db", "")
        self.repo = repository or KnowledgeRepository(
            Path(configured_db) if configured_db else None)
        self.curriculum = CurriculumEngine(
            self.repo, pilot_raaga=getattr(self.settings, "pilot_raaga",
                                           "Keeravani"))
        self.research = ResearchAgent(self.repo, self.library, self.settings, llm)
        self.practice = PracticeEngine(self.repo, self.library, self.settings)

        self.current_activity = "idle"
        self.last_step: Optional[LearningStep] = None
        self.history: List[LearningStep] = []
        self.errors: List[str] = []

        # Agent Factory (docs/PLAN_agent_factory.md, increment F2): opened
        # lazily on first use so an app or test that never trains never
        # touches factory.db.
        self._factory_store = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.RLock()
        self.on_progress = None            # optional callback for the UI

        self._restore()

    # ==================================================================
    # startup (specification section 9)
    # ==================================================================
    def _restore(self) -> None:
        version = self.repo.schema_version
        raaga = self.curriculum.current_raaga()
        # Write the focus down so a restart resumes the same study, even if
        # the creator never explicitly chose a raaga this session.
        self.repo.set_state("current_raaga", raaga)
        self.repo.set_state("last_session", time.time())
        self.repo.log_event("agent.started",
                            f"schema {version}, studying {raaga}", raaga=raaga)
        # A raaga the agent has never met still needs its reference book.
        if not self.repo.facts(raaga):
            written = self.research.seed_structural_knowledge(raaga)
            log.info("seeded %d structural facts for %s", written, raaga)
        pending = self.repo.pending_tasks()
        if pending:
            log.info("%d task(s) restored from the last session", len(pending))
        self.current_activity = "ready"

    # ==================================================================
    # knowledge for the composer
    # ==================================================================
    def raaga_for_composition(self, name: str) -> Tuple[Optional[Raaga], float]:
        """The raaga as the agent knows it, for generating music."""
        return learned_raaga(self.repo, self.library, name)

    def phrase_bank(self, name: str, min_confidence: float = 0.4
                    ) -> List[List[str]]:
        return learned_phrase_bank(self.repo, name, min_confidence)

    def phrase_index(self, name: str = "") -> PhraseIndex:
        return PhraseIndex.from_repository(self.repo, name)

    def evaluator(self, name: str = "") -> Evaluator:
        return Evaluator(self.library, self.phrase_index(name))

    def knowledge_report(self, name: str) -> str:
        return describe_knowledge(self.repo, name)

    def knowledge_confidence(self, name: str) -> Dict[str, float]:
        return knowledge_confidence(self.repo, name)

    # ==================================================================
    # Agent Factory (docs/PLAN_agent_factory.md, increment F2)
    # ==================================================================
    def factory_store(self):
        """The shared factory store, opened on first use.

        ``raagacomposer.factory.store`` is imported here, not at module
        scope: the core package may still be under construction while this
        adapter is built, and a module-level import would break every test
        that only exercises ``learn_step``.
        """
        if self._factory_store is None:
            from ..factory.store import FactoryStore
            path = getattr(self.settings, "factory_db", "")
            self._factory_store = FactoryStore(Path(path) if path else None)
        return self._factory_store

    def profile(self):
        """The agent's ``AgentProfile``: created once, then loaded and kept
        current (capabilities, current raaga, knowledge version) on every
        call."""
        from ..factory.models import AgentProfile

        store = self.factory_store()
        existing = next((p for p in store.profiles(domain="carnatic-music")
                         if p.name == "raga-agent"), None)
        raaga = self.curriculum.current_raaga()
        capabilities = sorted({
            (f"{u.curriculum_unit_id}:{u.raaga_name}" if u.raaga_name
             else u.curriculum_unit_id)
            for u in (self.curriculum.universal_units()
                     + self.curriculum.raaga_units(raaga))})
        knowledge_version = (f"schema{self.repo.schema_version}-"
                             f"curriculum{self.curriculum.version}")
        if existing is not None:
            existing.capabilities = capabilities
            existing.current_curriculum = raaga
            existing.knowledge_version = knowledge_version
            store.save_profile(existing)
            return existing
        profile = AgentProfile(
            name="raga-agent", role="student musician", domain="carnatic-music",
            capabilities=capabilities, current_curriculum=raaga,
            knowledge_version=knowledge_version)
        store.save_profile(profile)
        return profile

    def _record_cycle_in_curriculum(self, lesson, outcome) -> None:
        """The curriculum is the other ledger of the same study: a unit the
        student applied independently under the trainer is a unit passed,
        and a cycle that failed every test is a failed attempt, so
        ``next_unit`` moves on or rests exactly as it does for learn_step.
        Without this the trainer would set the first lesson forever."""
        from ..factory.models import TestLevel

        unit = self.curriculum.unit(lesson.origin) if lesson.origin else None
        if unit is None or not outcome.results:
            return
        if self.repo.progress(unit.id).status == "passed":
            return
        independent = [r for r in outcome.results
                       if r.passed and r.level >= TestLevel.T4_INDEPENDENT_APPLICATION]
        if independent:
            best = max(r.score for r in independent)
            self.curriculum.record_attempt(
                unit, best, True,
                f"factory: {outcome.mastery_after.label} after "
                f"{len(outcome.results)} test(s)")
        elif not any(r.passed for r in outcome.results):
            worst = min(r.score for r in outcome.results)
            self.curriculum.record_attempt(
                unit, worst, False,
                f"factory: every test failed at {outcome.mastery_after.label}")

    def train_step(self, raaga: Optional[str] = None):
        """One turn of the Agent Factory learning cycle (document 01's ten
        steps), on top of the same curriculum, knowledge and engines
        ``learn_step`` uses.  Returns ``None`` when the curriculum is
        complete for the current raaga, exactly like ``learn_step`` does for
        its own loop."""
        from ..factory.cycle import LearningCycle
        from .rules import hard_rules
        from .student import RagaStudent
        from .trainer import RagaTrainer

        with self._lock:
            if raaga:
                self.curriculum.set_current_raaga(raaga)
            active = self.curriculum.current_raaga()
            store = self.factory_store()
            profile = self.profile()
            trainer = RagaTrainer(self, store)
            lesson = trainer.next_lesson(profile, store.results(profile.id))
            if lesson is None:
                return None
            student = RagaStudent(self, profile)
            rules = hard_rules(self.library, self.phrase_index(active))
            cycle = LearningCycle(store, student, trainer, rules=rules)
            outcome = cycle.run(lesson)
            self._record_cycle_in_curriculum(lesson, outcome)
            self.repo.log_event(
                "factory.cycle",
                f"{lesson.concept}: {outcome.mastery_before.label} -> "
                f"{outcome.mastery_after.label}"
                f"{' (advanced)' if outcome.advanced else ''}",
                raaga=active)
            return outcome

    # ==================================================================
    # a failed attempt becomes a lesson (section 38)
    # ==================================================================
    def record_lessons(self, evaluation: Evaluation, *, raaga: str,
                       unit_id: str = "", attempt: int = 0, task: str = "",
                       method: str = "", result: float = 0.0,
                       source_run: str = "",
                       confidence: Optional[float] = None,
                       findings: Optional[Sequence[Finding]] = None
                       ) -> List[Lesson]:
        """Turn every finding on a failed evaluation into stored knowledge.

        Each kind of finding becomes one lesson; a finding of a kind already
        on record for this raaga and unit strengthens it instead of piling
        up a duplicate (``KnowledgeRepository.add_lesson``).  Within one
        attempt the same kind counts once, however many exercises raised it,
        so recurrences count attempts, not exercises.  No findings means
        nothing is written.  ``findings`` overrides the evaluation's own
        list when an attempt was marked on several evaluations.
        """
        stored: List[Lesson] = []
        seen: set = set()
        chosen = list(findings if findings is not None
                      else evaluation.findings)
        # Every phrase the attempt was caught copying, from every exercise:
        # the evidence of a not_original finding starts with the phrase id.
        # The evaluation's own originality report names only the last
        # exercise's match, which is not always the one that matters.
        copied: List[str] = []
        for finding in chosen:
            if finding.kind == "not_original":
                head = (finding.evidence or "").split(" ")[0]
                if head.startswith("phr_") and head not in copied:
                    copied.append(head)
        if (evaluation.originality and evaluation.originality.matched_phrase_id
                and evaluation.originality.matched_phrase_id not in copied):
            copied.append(evaluation.originality.matched_phrase_id)
        for finding in chosen:
            if finding.kind in seen:
                continue
            seen.add(finding.kind)
            related: List[str] = []
            if finding.kind == "not_original":
                related = list(copied)
            base_confidence = (confidence if confidence is not None
                              else evaluation.confidence)
            lesson = Lesson(
                raaga=raaga, unit_id=unit_id, attempt=attempt, task=task,
                method=method, result=result, kind=finding.kind,
                dimension=finding.dimension, failure_reason=finding.text,
                evidence=finding.evidence, correction=evaluation.recommendation,
                related=related, source_run=source_run,
                confidence=min(1.0, base_confidence * finding.weight))
            recorded, _ = self.repo.add_lesson(lesson)
            stored.append(recorded)
        return stored

    # ==================================================================
    # Apply Brief: knowledge-driven raaga suggestion (section 10)
    # ==================================================================
    def suggest_raagas(self, brief: CreativeBrief, limit: int = 4
                       ) -> List[RaagaSuggestion]:
        words = expand_feel_words(brief.mood, brief.feel, brief.situation,
                                  brief.notes, brief.song_type, brief.vocal_feel)
        explicit = self.library.get(brief.raaga_preference) if \
            brief.raaga_preference else None
        if explicit is None:
            explicit = self.library.find_in_text(
                " ".join((brief.feel, brief.notes, brief.situation)))

        studied = {r for r in self.repo.known_raagas()}
        suggestions: List[RaagaSuggestion] = []

        for raaga in self.library.all():
            evidence: List[str] = []
            matched = [m for m in raaga.moods if m in words]
            score = float(len(matched))
            confidence = 0.35
            learned = raaga.name in studied

            if learned:
                facets = knowledge_confidence(self.repo, raaga.name)
                confidence = 0.35 + 0.5 * facets["overall"]
                if facets["phrases"]:
                    evidence.append(f"{facets['phrases']} phrases heard")
                    score += min(1.5, facets["phrases"] / 8.0)
                mood_fact = self.repo.best_fact(raaga.name, "moods")
                if mood_fact:
                    learned_moods = [m.strip().lower()
                                     for m in mood_fact.value.split(",")]
                    extra = [m for m in learned_moods if m in words]
                    if extra:
                        score += 0.5 * len(extra)
                        evidence.append("mood learned from " +
                                        (self.repo.source(mood_fact.source_id).title
                                         if self.repo.source(mood_fact.source_id)
                                         else "memory"))
                tempo_fact = self.repo.best_fact(raaga.name, "observed_tempo")
                if tempo_fact and brief.tempo_preference:
                    try:
                        heard = float(tempo_fact.value)
                        if abs(heard - brief.tempo_preference) < 20:
                            score += 0.4
                            evidence.append(f"heard played near {heard:.0f} bpm")
                    except ValueError:
                        pass

            if brief.tempo_preference and raaga.tempo_range:
                low, high = raaga.tempo_range[0], raaga.tempo_range[-1]
                score += 0.6 if low <= brief.tempo_preference <= high else -0.3

            if raaga is explicit:
                score += 10.0
                confidence = max(confidence, 0.9)
                evidence.append("you asked for it")

            if score <= 0:
                continue
            if matched:
                confidence = min(0.95, confidence + 0.08 * len(matched))
            reason = self._reason(raaga, matched, raaga is explicit, learned)
            suggestions.append(RaagaSuggestion(
                name=raaga.name, score=round(score, 3),
                confidence=round(min(0.95, confidence), 3), reason=reason,
                evidence=evidence, learned=learned))

        if not suggestions:
            fallback = self.library.get("Mohanam") or self.library.all()[0]
            suggestions = [RaagaSuggestion(
                name=fallback.name, score=0.1, confidence=0.25,
                reason="The brief is still thin, so this is a safe, open-sounding "
                       "starting point rather than a considered choice.",
                learned=fallback.name in studied)]

        suggestions.sort(key=lambda s: (-s.score, -s.confidence, s.name))
        self.repo.log_event(
            "brief.suggested",
            f"{', '.join(s.name for s in suggestions[:limit])} for "
            f"{brief.mood}/{brief.feel}"[:180])
        return suggestions[:limit]

    @staticmethod
    def _reason(raaga: Raaga, matched: Sequence[str], explicit: bool,
                learned: bool) -> str:
        if explicit:
            return f"You asked for {raaga.name}. {raaga.character()}".strip()
        head = f"Carries {', '.join(matched[:3])}. " if matched else ""
        tail = raaga.character()
        studied = " I have studied this one." if learned else \
            " I have not studied this one yet, so this is from the reference book."
        return (head + tail + studied).strip()

    # ==================================================================
    # the learning loop (section 15)
    # ==================================================================
    def learn_step(self, raaga: Optional[str] = None) -> LearningStep:
        """One turn of the loop: study something, or find material to study."""
        with self._lock:
            name = raaga or self.curriculum.current_raaga()
            unit = self.curriculum.next_unit(name)
            if unit is None:
                step = LearningStep(action="idle", raaga=name,
                                    detail=f"nothing left to study for {name}")
                return self._finish_step(step)

            self.current_activity = f"studying {unit.id}"
            needs_material = (unit.retry_policy == "ingest_more_sources"
                              or unit.skill_type == "recall.phrases")
            if needs_material and not self._enough_material(unit, name):
                step = self._research_step(unit, name)
                if step.passed:
                    return self._finish_step(step)
                # Nothing new to listen to. Attempt the lesson anyway rather
                # than asking for material over and over: the attempt is
                # scored, the unit eventually rests, and the creator is told
                # what would help.
                practice = self._practice_step(unit, name)
                practice.detail = f"{step.detail}; tried anyway - {practice.detail}"
                return self._finish_step(practice)
            return self._finish_step(self._practice_step(unit, name))

    def _enough_material(self, unit: Unit, raaga: str) -> bool:
        wanted = int(unit.params.get("min_phrases",
                                     unit.minimum_examples_required))
        have = len(self.repo.phrases(
            raaga=raaga,
            min_confidence=float(unit.params.get("min_confidence", 0.4)),
            limit=500))
        return have >= wanted

    def _research_step(self, unit: Unit, raaga: str) -> LearningStep:
        step = LearningStep(action="research", unit_id=unit.id, raaga=raaga)
        limit = int(getattr(self.settings, "learning_max_sources_per_lesson", 4))
        candidates = self.research.find_sources(raaga, unit.learning_goal, limit)
        if not candidates:
            step.detail = (f"no new permitted material for {raaga}; "
                           f"add audio to the learning folder to teach it more")
            self.repo.log_event("research.exhausted", step.detail, unit_id=unit.id,
                                raaga=raaga)
            return step

        learned = rejected = 0
        used: List[str] = []
        for candidate in candidates:
            result = self.research.ingest(candidate)
            if result.error and not result.analysed:
                self.errors.append(f"{candidate.title}: {result.error}")
                continue
            learned += result.phrases_learned
            rejected += result.phrases_rejected
            used.append(candidate.title)
        step.detail = (f"listened to {len(used)} source(s) for {raaga}: "
                       f"{learned} phrase(s) learned, {rejected} rejected")
        step.score = 1.0 if learned else 0.0
        step.passed = learned > 0
        return step

    def _practice_step(self, unit: Unit, raaga: str) -> LearningStep:
        step = LearningStep(action="practice", unit_id=unit.id, raaga=raaga)
        # Each attempt sets fresh exercises, and the same attempt sets the
        # same ones in every run.  Seeding from the clock replayed one failed
        # attempt identically for as long as the second lasted (REG-100).
        attempt = self.repo.progress(unit.id).attempts
        seed = practice_seed(unit.id, attempt)
        # What the agent's own lessons say before this attempt runs
        # (spec section 15 #9, docs/PLAN_learning_loop.md Part 1).
        guidance = build_guidance(self.repo, raaga, unit.id)
        report = self.practice.run(unit, raaga, seed=seed, guidance=guidance)
        progress = self.curriculum.record_attempt(
            unit, report.score, report.passed, report.detail)
        step.score = report.score
        step.passed = report.passed
        step.detail = (f"{unit.learning_goal} -> {report.score:.2f} "
                       f"({'passed' if report.passed else 'retry'})")
        if not report.passed and report.evaluation is not None:
            step.detail += f"; {report.evaluation.recommendation}"
        if not guidance.is_empty():
            step.detail += f"; guided: {guidance.describe()}"
            self.repo.log_event("guidance.applied", ", ".join(guidance.kinds),
                                unit_id=unit.id, raaga=raaga)
        if report.passed and report.artifacts:
            self._keep_best_artifact(unit, raaga, report)
        if progress.status == "failed":
            step.detail += " - giving this unit a rest after repeated failures"
        if not report.passed:
            method = f"{unit.skill_type} seed {seed}"
            source_run = f"practice:{unit.id}:{attempt}"
            if report.evaluation is not None:
                lessons = self.record_lessons(
                    report.evaluation, raaga=raaga, unit_id=unit.id,
                    attempt=attempt, task=unit.learning_goal, method=method,
                    result=report.score, source_run=source_run,
                    findings=report.findings or None)
            else:
                lessons = self._record_exercise_lessons(
                    unit, raaga, attempt, method, source_run, report)
            self._note_recurrence(step, lessons)
        elif not guidance.is_empty():
            self._apply_lessons_of_unit(unit, raaga, guidance)
        return step

    def _apply_lessons_of_unit(self, unit: Unit, raaga: str,
                               guidance: Guidance) -> None:
        """A pass retires the lessons of this unit that guided it - they did
        their job.  Lessons from the raaga's other units stay on record: the
        same mistake elsewhere still needs avoiding (docs/PLAN_learning_loop.md,
        "Lessons are per raaga with a unit bonus, not per unit only")."""
        unit_lesson_ids = {l.id for l in
                           self.repo.lessons(raaga=raaga, unit_id=unit.id,
                                             limit=500)}
        applied = [lid for lid in guidance.lesson_ids if lid in unit_lesson_ids]
        for lesson_id in applied:
            self.repo.mark_lesson_applied(lesson_id)
        self.repo.log_event("lessons.applied", str(len(applied)),
                            unit_id=unit.id, raaga=raaga)

    @staticmethod
    def _exercise_base_name(name: str) -> str:
        """"name the swara 3" -> "name the swara": the same exercise, retried."""
        return re.sub(r"\s+\d+$", "", name).strip()

    def _record_exercise_lessons(self, unit: Unit, raaga: str, attempt: int,
                                 method: str, source_run: str,
                                 report: PracticeReport) -> List[Lesson]:
        """A quiz-style attempt has no Evaluation; its failed exercises are
        the findings instead, grouped so a retried exercise makes one lesson."""
        groups: Dict[str, Any] = {}
        for exercise in report.exercises:
            if exercise.passed:
                continue
            base = self._exercise_base_name(exercise.name)
            groups.setdefault(base, exercise)
        stored: List[Lesson] = []
        for base, exercise in groups.items():
            lesson = Lesson(
                raaga=raaga, unit_id=unit.id, attempt=attempt,
                task=unit.learning_goal, method=method, result=report.score,
                kind=f"exercise:{base}", dimension=unit.skill_type,
                failure_reason=f"expected {exercise.expected}, "
                              f"heard {exercise.heard}",
                evidence=exercise.detail, correction="",
                source_run=source_run, confidence=0.6)
            recorded, _ = self.repo.add_lesson(lesson)
            stored.append(recorded)
        return stored

    @staticmethod
    def _note_recurrence(step: LearningStep, lessons: Sequence[Lesson]) -> None:
        """Make section 38's "no rediscovery" visible: say so when it fires."""
        recurring = [l for l in lessons if l.recurrences > 1]
        if not recurring:
            return
        worst = max(recurring, key=lambda l: l.recurrences)
        step.detail += f"; again: {worst.kind} (x{worst.recurrences})"

    def _keep_best_artifact(self, unit: Unit, raaga: str,
                            report: PracticeReport) -> None:
        """Good practice output becomes material the composer can draw on."""
        best = max(report.artifacts, key=len, default=None)
        if not best or len(best) < 3:
            return
        swaras = [n.swara for n in best]
        index = self.phrase_index(raaga)
        if not check_originality(swaras, index).is_original:
            return
        phrase = Phrase(
            raaga=raaga, swaras=swaras, midi=[n.midi for n in best],
            durations=[round(n.duration, 3) for n in best],
            function="practice", source_id="", confidence=0.5,
            notes=f"the agent's own practice for {unit.id}")
        _, is_new = self.repo.add_phrase(phrase)
        if is_new:
            self.repo.log_event("practice.kept",
                                f"kept an original phrase from {unit.id}",
                                unit_id=unit.id, raaga=raaga)

    def _finish_step(self, step: LearningStep) -> LearningStep:
        self.last_step = step
        self.history.append(step)
        del self.history[:-200]
        self.repo.set_state("last_step", {
            "at": step.at, "action": step.action, "unit": step.unit_id,
            "raaga": step.raaga, "detail": step.detail})
        self.current_activity = "idle"
        if self.on_progress:
            try:
                self.on_progress(step)
            except Exception:  # noqa: BLE001
                log.debug("progress callback failed", exc_info=True)
        return step

    def learn(self, cycles: int = 1, raaga: Optional[str] = None
              ) -> List[LearningStep]:
        return [self.learn_step(raaga) for _ in range(max(1, cycles))]

    def learn_until(self, unit_id: str, max_steps: int = 40) -> List[LearningStep]:
        """Study until a particular unit passes, or the budget runs out."""
        steps: List[LearningStep] = []
        for _ in range(max_steps):
            if self.repo.progress(unit_id).status == "passed":
                break
            step = self.learn_step()
            steps.append(step)
            if step.action == "idle":
                break
        return steps

    def study_raaga(self, name: str) -> str:
        """"Learn Keeravani." - switch the deep-learning focus."""
        raaga = self.library.get(name)
        if raaga is None:
            return f"I do not have {name} in my library, so I cannot study it."
        self.curriculum.set_current_raaga(raaga.name)
        if not self.repo.facts(raaga.name):
            self.research.seed_structural_knowledge(raaga.name)
        return (f"Now studying {raaga.name}. "
                f"Next: {self.curriculum.stage_summary(raaga.name)['next_goal']}")

    # ==================================================================
    # background learning (section 16)
    # ==================================================================
    @property
    def is_learning(self) -> bool:
        return self._thread is not None and self._thread.is_alive() \
            and not self._pause.is_set()

    @property
    def is_paused(self) -> bool:
        return self._thread is not None and self._pause.is_set()

    def start_learning(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            self._pause.clear()
            return True
        self._stop.clear()
        self._pause.clear()
        self._thread = threading.Thread(target=self._background, daemon=True,
                                        name="raaga-learning")
        self._thread.start()
        self.repo.log_event("learning.started", "background learning started")
        return True

    def pause_learning(self) -> None:
        self._pause.set()
        self.repo.log_event("learning.paused", "")

    def resume_learning(self) -> None:
        self._pause.clear()
        self.repo.log_event("learning.resumed", "")

    def stop_learning(self, wait: bool = True) -> None:
        self._stop.set()
        self._pause.clear()
        thread, self._thread = self._thread, None
        if thread and wait and thread.is_alive():
            thread.join(timeout=15.0)
            if thread.is_alive():
                log.error("the learning thread did not stop in time")
        if not self.repo.closed:
            self.repo.log_event("learning.stopped", "")

    def _background(self) -> None:
        max_steps = int(getattr(self.settings, "learning_max_steps_per_session", 200))
        pause_seconds = float(getattr(self.settings, "learning_step_pause", 0.5))
        done = 0
        while not self._stop.is_set() and done < max_steps:
            if self._pause.is_set():
                self._stop.wait(0.2)
                continue
            try:
                step = self.learn_step()
            except Exception as exc:  # noqa: BLE001
                self.errors.append(str(exc))
                log.exception("background learning failed")
                time.sleep(1.0)
                continue
            done += 1
            # Interruptible waits: a stop request must be honoured promptly so
            # the thread is never still querying memory when it is closed.
            self._stop.wait(2.0 if step.action == "idle" else pause_seconds)
        self.current_activity = "idle"

    # ==================================================================
    # feedback and explanation (sections 14, 3.1)
    # ==================================================================
    @staticmethod
    def feedback_sentiment(text: str) -> str:
        """"negative" / "positive" / "neutral", by the same rules everywhere."""
        if NEGATIVE.search(text or ""):
            return "negative"
        if POSITIVE.search(text or ""):
            return "positive"
        return "neutral"

    @staticmethod
    def feedback_kinds(text: str) -> List[str]:
        """The finding kinds the creator's words point at, so a complaint
        about a tune the critic passed still becomes guidance the next tune
        obeys ("too mechanical" -> vary more, add gamaka)."""
        low = (text or "").lower()
        kinds: List[str] = []
        for pattern, mapped in FEEDBACK_KINDS:
            if pattern.search(low):
                for kind in mapped:
                    if kind not in kinds:
                        kinds.append(kind)
        return kinds

    def record_feedback(self, text: str, raaga: str = "",
                        swaras: Optional[Sequence[str]] = None,
                        target_kind: str = "composition",
                        target_id: str = "") -> str:
        """Take a correction seriously: store it and act on it."""
        sentiment = self.feedback_sentiment(text)
        named = self.library.find_in_text(text or "")
        raaga = raaga or (named.name if named else self.curriculum.current_raaga())
        feedback_id = self.repo.add_feedback(
            target_kind=target_kind, target_id=target_id, text=text,
            sentiment=sentiment, raaga=raaga)

        if sentiment == "negative":
            weakened = self._weaken_phrases(raaga, swaras)
            self.repo.mark_feedback_applied(feedback_id)
            if weakened:
                return (f"Noted. I have lowered my confidence in {weakened} "
                        f"phrase(s) I had learned for {raaga}, so I will lean on "
                        f"them less.")
            return (f"Noted against {raaga}. I had nothing learned to blame, so "
                    f"I have recorded it and will weigh it when I next study.")
        if sentiment == "positive":
            strengthened = self._strengthen_phrases(raaga, swaras)
            self.repo.mark_feedback_applied(feedback_id)
            if strengthened:
                return (f"Thank you - I have raised my confidence in "
                        f"{strengthened} phrase(s) for {raaga}.")
            return "Thank you, noted."
        return "Noted."

    def _weaken_phrases(self, raaga: str, swaras: Optional[Sequence[str]],
                        amount: float = 0.25) -> int:
        phrases = self.repo.phrases(raaga=raaga, limit=200)
        if not phrases:
            return 0
        targets: List[Phrase] = []
        if swaras:
            index = PhraseIndex(n=3)
            index.add_many(phrases)
            run, phrase_id, _ = index.longest_shared_run(swaras)
            if phrase_id and run >= 3:
                found = self.repo.phrase(phrase_id)
                if found:
                    targets = [found]
        if not targets:
            # Nothing specific to blame: doubt the least certain thing first.
            targets = sorted(phrases, key=lambda p: p.confidence)[:2]
        for phrase in targets:
            new_confidence = max(0.0, phrase.confidence - amount)
            if new_confidence < 0.15:
                self.repo.reject_phrase(
                    phrase.id, "the creator said this does not sound right")
            else:
                self.repo.set_phrase_confidence(
                    phrase.id, new_confidence,
                    "confidence lowered after the creator's correction")
        self.repo.log_event("feedback.applied",
                            f"lowered confidence on {len(targets)} phrase(s)",
                            raaga=raaga)
        return len(targets)

    def _strengthen_phrases(self, raaga: str, swaras: Optional[Sequence[str]],
                            amount: float = 0.1) -> int:
        if not swaras:
            return 0
        phrases = self.repo.phrases(raaga=raaga, limit=200)
        if not phrases:
            return 0
        index = PhraseIndex(n=3)
        index.add_many(phrases)
        run, phrase_id, _ = index.longest_shared_run(swaras)
        if not phrase_id or run < 3:
            return 0
        phrase = self.repo.phrase(phrase_id)
        if phrase is None:
            return 0
        self.repo.set_phrase_confidence(
            phrase.id, min(0.99, phrase.confidence + amount),
            "the creator liked this")
        return 1

    def explain(self, question: str, raaga: str = "") -> str:
        """Answer questions about what it knows and why it chose something."""
        raaga = raaga or self.curriculum.current_raaga()
        low = (question or "").lower()
        named = self.library.find_in_text(question or "")
        if named:
            raaga = named.name

        if "arohanam" in low or "avarohanam" in low or "scale" in low:
            up = self.repo.best_fact(raaga, "arohanam")
            down = self.repo.best_fact(raaga, "avarohanam")
            if not up:
                return f"I have not learned the scale of {raaga} yet."
            return (f"{raaga}\n  arohanam:   {up.value}\n"
                    f"  avarohanam: {down.value if down else '(not learned)'}\n"
                    f"  I learned this from "
                    f"{self._source_name(up.source_id)} "
                    f"(confidence {up.confidence:.2f}).")
        if "phrase" in low or "prayoga" in low or "why" in low:
            phrases = self.repo.phrases(raaga=raaga, limit=5)
            if not phrases:
                return (f"I have not heard any phrases of {raaga} yet, so I am "
                        f"composing from its scale alone.")
            rows = [f"I lean on these {raaga} phrases, most trusted first:"]
            for phrase in phrases:
                rows.append(f"  {' '.join(phrase.swaras):<26} "
                            f"confidence {phrase.confidence:.2f}, heard in "
                            f"{self._source_name(phrase.source_id)}")
            return "\n".join(rows)
        if "learn" in low or "progress" in low or "studying" in low:
            summary = self.curriculum.stage_summary(raaga)
            return (f"I am at stage {summary['stage']} studying {raaga}. "
                    f"Foundations {summary['foundations']}, raaga units "
                    f"{summary['raaga_units']}. Next: {summary['next_goal']}")
        if "know" in low:
            return describe_knowledge(self.repo, raaga)
        return describe_knowledge(self.repo, raaga)

    def _source_name(self, source_id: str) -> str:
        source = self.repo.source(source_id) if source_id else None
        return source.title if source else "my own practice"

    # ==================================================================
    # composition experience
    # ==================================================================
    def record_composition(self, *, project_id: str, title: str, raaga: str,
                           brief: CreativeBrief, notes: Sequence[Note],
                           structure: Optional[Dict[str, Any]] = None,
                           tempo_bpm: float = 0.0, seed: int = 0
                           ) -> Tuple[str, Evaluation]:
        """Mark the agent's own work and remember how it went.

        A tune is also the T10 (real-world) test of the Agent Factory ladder
        (docs/PLAN_agent_factory.md, "Item 4, integrated"): every composition
        files a ``TestSpec``/``TestResult`` and applies mastery evidence to
        ``compose:<raaga>``.  Guarded so a factory failure never blocks a
        tune - the composer must keep working even if the factory database
        is unavailable.
        """
        raaga_view, _ = self.raaga_for_composition(raaga)
        target = raaga_view or self.library.get(raaga)
        evaluation = Evaluation()
        if target is not None and notes:
            evaluation = self.evaluator(raaga).evaluate(
                notes, target, brief=brief, tempo_bpm=tempo_bpm,
                learned_phrases=self.phrase_bank(raaga))
        composition_id = self.repo.record_composition(
            project_id=project_id, title=title, raaga=raaga,
            brief={"mood": brief.mood, "feel": brief.feel,
                   "situation": brief.situation, "language": brief.language},
            structure=structure or {},
            scores=evaluation.scores, final_score=evaluation.overall(),
            notes=evaluation.recommendation)
        try:
            version = (structure or {}).get("version", 0)
            self._record_t10_composition(project_id, version, raaga, notes,
                                         evaluation, seed)
        except Exception as exc:  # noqa: BLE001 - the factory must never block a tune
            log.warning("could not file the composition as a T10 test: %s", exc)
        return composition_id, evaluation

    def _record_t10_composition(self, project_id: str, version: Any, raaga: str,
                                notes: Sequence[Note], evaluation: Evaluation,
                                seed: int) -> None:
        """Generate Tune is the T10 test (docs/PLAN_agent_factory.md, "Item 4,
        integrated"): the student's claim is its own self-evaluation (the
        learned view); the trainer's claim is the same verdict read off the
        library view, so a tune that only looks good through what the agent
        has heard does not pass on its own say-so."""
        from ..factory.mastery import apply_evidence
        from ..factory.models import Split, TestLevel, TestResult, TestSpec

        store = self.factory_store()
        profile = self.profile()
        concept = f"compose:{raaga}"
        threshold = float(getattr(self.settings, "compose_threshold", 0.7))
        overall = evaluation.overall()
        passed = overall >= threshold
        student_verdict = "acceptable" if passed else "needs work"

        library_raaga = self.library.get(raaga)
        trainer_evaluation = evaluation
        if library_raaga is not None and notes:
            trainer_evaluation = Evaluator(
                self.library, self.phrase_index(raaga)).evaluate(notes, library_raaga)
        trainer_verdict = ("acceptable" if trainer_evaluation.overall() >= threshold
                           else "needs work")

        test = TestSpec(capability=concept, level=TestLevel.T10_REAL_WORLD,
                        split=Split.REAL_WORLD, objective=False, ambiguity=0.6,
                        seed=seed, payload={"project_id": project_id,
                                            "version": version})
        store.save_test(test)

        findings = evaluation.findings
        result = TestResult(
            test_id=test.id, agent_id=profile.id, level=TestLevel.T10_REAL_WORLD,
            split=Split.REAL_WORLD, score=overall, passed=passed,
            student_claim=student_verdict, student_confidence=evaluation.confidence,
            trainer_claim=trainer_verdict,
            trainer_confidence=trainer_evaluation.confidence,
            failure_mode=findings[0].kind if findings else "",
            evidence=[f.text for f in findings])
        store.save_result(result)

        record = store.mastery(profile.id, concept)
        apply_evidence(record, "T10", result.id, passed=passed)
        store.save_mastery(record)

    def explain_choice(self, melody, raaga: str) -> str:
        """"Why did you choose this phrase?" (spec section 16), answered from
        what the tune actually borrowed (``melody.provenance``), what a
        guided rewrite avoided (``melody.validation`` and
        ``melody.guidance_note``), and what is still an open lesson for the
        raaga."""
        rows: List[str] = []
        for entry in list(melody.provenance)[:5]:
            when = self._time_at_note(melody, int(entry.get("start", 0)))
            origin = entry.get("origin") or "the raaga library"
            rows.append(f"at {when} I quoted {entry.get('swaras', '')} "
                       f"from {origin}")
        if not rows:
            rows.append("I did not quote a particular phrase here - this "
                       "line is my own, from the raaga's scale and what I "
                       "have practised.")
        rewrites = [line for line in melody.validation if line.startswith("rewrite")]
        rows.extend(rewrites)
        guidance_note = getattr(melody, "guidance_note", "")
        if guidance_note:
            rows.append(f"I was guided to: {guidance_note}")
        open_lessons = [l for l in self.repo.lessons(raaga=raaga, limit=50)
                        if not l.applied]
        remarks = [l for l in open_lessons if l.kind == "creator_feedback"]
        if remarks:
            rows.append(f"I kept your note in mind: "
                        f"\"{remarks[0].failure_reason[:120]}\"")
        acted = [l for l in open_lessons if l.kind != "creator_feedback"]
        if acted:
            worst = max(acted, key=lambda l: l.recurrences)
            what = {
                "repetitive": "varied the line more",
                "no_gamaka": "added ornament",
                "too_many_leaps": "moved by step",
                "no_cadence": "closed on a resting note",
                "neighbour_drift": "leaned on the raaga's own notes",
                "no_idiom": "quoted the raaga's own phrases more",
                "off_beat": "kept to the beat",
                "not_original": "avoided a phrase I had copied",
            }.get(worst.kind, f"worked on {worst.kind.replace('_', ' ')}")
            source = ("your feedback" if worst.dimension == "creator"
                      else "my own critique")
            rows.append(f"I {what} because of {source}: {worst.kind}, "
                        f"seen {worst.recurrences} time"
                        f"{'s' if worst.recurrences != 1 else ''}")
        return "\n".join(rows)

    @staticmethod
    def _time_at_note(melody, index: int) -> str:
        seconds = melody.notes[index].start if 0 <= index < len(melody.notes) else 0.0
        return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"

    def record_field_feedback(self, raaga: str, text: str, positive: bool,
                              evaluation: Optional[Evaluation],
                              project_id: str) -> Optional[Any]:
        """Creator feedback on a tune is field evidence (docs/PLAN_agent_factory.md,
        "Item 4, integrated"; document 02 section 5: the creator is the
        external evidence that decides).  Negative feedback opens a dispute
        between the agent's own verdict and the creator's, resolves it at
        once in the creator's favour, and leaves a candidate reusable lesson
        and a failed mastery signal.  Positive feedback is field evidence
        (L9) and a passed real-world result.  Returns the ``Ruling`` when one
        was made, else ``None``."""
        from ..factory.mastery import apply_evidence
        from ..factory.models import (Dispute, KnowledgeClass, MasteryLevel,
                                      ReusableLesson, Ruling, Split, TestLevel,
                                      TestResult, TestSpec, ValidationStatus)

        store = self.factory_store()
        profile = self.profile()
        concept = f"compose:{raaga}"
        threshold = float(getattr(self.settings, "compose_threshold", 0.7))
        evaluation = evaluation or Evaluation()
        overall = evaluation.overall()
        findings = evaluation.findings
        student_verdict = "acceptable" if overall >= threshold else "needs work"

        if not positive:
            dispute = Dispute(
                agent_id=profile.id,
                question=f"is this tune acceptable in {raaga}?",
                student_claim=student_verdict, trainer_claim="not acceptable",
                evidence_student=[f.text for f in findings],
                evidence_trainer=[text],
                student_confidence=evaluation.confidence, trainer_confidence=0.95,
                critical=False)
            store.save_dispute(dispute)

            # The guidance kinds the last evaluation's findings map to - what
            # a Guidance built from just this tune's own findings would
            # correct, not the raaga's whole lesson history.
            guidance_kinds = sorted({f.kind for f in findings}
                                    | set(self.feedback_kinds(text)))
            ruling = Ruling(
                dispute_id=dispute.id, ruling="the creator's verdict stands",
                accepted_claim="trainer", rejected_claim="student",
                rationale=text, correction_student=", ".join(guidance_kinds),
                confidence=0.95, needs_external_evidence=False,
                decided_by="creator")
            store.save_ruling(ruling)

            reusable = ReusableLesson(
                source_event=ruling.id,
                rule_or_procedure=f"the creator rejected a {raaga} tune: {text}",
                knowledge_class=KnowledgeClass.EXPERIENCE,
                validation_status=ValidationStatus.CANDIDATE,
                scope_domain="carnatic-music", scope_concept=concept,
                source_agent_id=profile.id, confidence=0.6)
            store.save_reusable(reusable)

            record = store.mastery(profile.id, concept)
            apply_evidence(record, "T10", ruling.id, passed=False)
            store.save_mastery(record)
            return ruling

        # Positive feedback: field evidence (document 05 section 6).  One
        # word of praise validates a capability the agent has already shown
        # it generalises (L7 from real-world passes); before that it counts
        # as one more real-world pass, not expertise (document 04 section
        # 5: one success is not a rule).
        record = store.mastery(profile.id, concept)
        evidence_id = f"feedback:{int(time.time() * 1000)}:{text[:40]}"
        kind = ("field" if record.level >= MasteryLevel.L7_GENERALIZES
                else "T10")
        apply_evidence(record, kind, evidence_id, passed=True)
        store.save_mastery(record)
        # Filed against a test of its own, so the History table can name the
        # raaga and the tune the praise was about.
        spec = TestSpec(
            capability=concept, level=TestLevel.T10_REAL_WORLD,
            split=Split.REAL_WORLD, novelty=1.0, difficulty=1.0, ambiguity=0.7,
            objective=False, expected="the creator accepts the tune",
            payload={"project_id": project_id, "feedback": text,
                     "raaga": raaga})
        store.save_test(spec)
        result = TestResult(
            test_id=spec.id, agent_id=profile.id, level=TestLevel.T10_REAL_WORLD,
            split=Split.REAL_WORLD, score=overall, passed=True,
            student_claim=student_verdict,
            student_confidence=evaluation.confidence, trainer_claim="acceptable",
            trainer_confidence=0.95, evidence=[text])
        store.save_result(result)
        return None

    # ==================================================================
    # status for the UI (section 17)
    # ==================================================================
    def status(self) -> Dict[str, Any]:
        raaga = self.curriculum.current_raaga()
        summary = self.curriculum.stage_summary(raaga)
        stats = self.repo.stats()
        facets = knowledge_confidence(self.repo, raaga)
        return {
            "activity": self.current_activity,
            "learning": self.is_learning,
            "paused": self.is_paused,
            "stage": summary["stage"],
            "current_raaga": raaga,
            "next_unit": summary["next_unit"],
            "next_goal": summary["next_goal"],
            "foundations": summary["foundations"],
            "raaga_units": summary["raaga_units"],
            "overall_percent": summary["overall_percent"],
            "mastered_raagas": summary["mastered_raagas"],
            "mastery": facets["overall"],
            "phrases": stats["phrases"],
            "sources": stats["sources"],
            "sources_analysed": stats["sources_analysed"],
            "facts": stats["facts"],
            "disputed_facts": stats["disputed_facts"],
            "compositions": stats["compositions"],
            "repository": stats["path"],
            "repository_bytes": stats["size_bytes"],
            "last_step": self.last_step.summary() if self.last_step else "",
            "errors": self.errors[-5:],
        }

    def recent_events(self, limit: int = 30) -> List[Dict[str, Any]]:
        return self.repo.events(limit)

    def close(self) -> None:
        # The learner must be stopped and joined *before* memory is closed:
        # a query running against a closed connection takes the process down.
        self.stop_learning(wait=True)
        if self._factory_store is not None and not self._factory_store.closed:
            try:
                self._factory_store.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("could not close the factory store: %s", exc)
        if self.repo.closed:
            return
        try:
            self.repo.set_state("last_session", time.time())
        except Exception as exc:  # noqa: BLE001
            log.warning("could not record the session end: %s", exc)
        self.repo.close()
