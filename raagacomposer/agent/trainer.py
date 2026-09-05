"""RagaTrainer: the domain half of the Trainer role (document 02).

Builds framework ``Lesson``s from curriculum units and the knowledge base,
generates and grades tests along the ladder (document 03), and remediates or
asks for harder variants as the agent's mastery grows.  No model call:
everything here reads stored knowledge and drives the existing evaluator and
practice engines.

``factory.mastery`` and ``factory.models.MasteryRecord`` are imported inside
the methods that need them: the core package may not exist yet while this
adapter is being built, and importing at module scope would make every
caller of this module fail before the core lands.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from ..factory.models import (AgentProfile, KnowledgeClass, Lesson,
                              Reiteration, ReiterationCheck, Remediation,
                              Split, TestLevel, TestResult, TestSpec)
from ..raaga.library import parse_swara
from ..training.lessons import is_stated
from .curriculum import Unit
from .evaluator import Evaluator
from .learned import CORE_KEYS
from .practice import TONIC, practice_seed
from .rules import _base_tokens, _reference_value
from .student import MEANINGS

if TYPE_CHECKING:  # pragma: no cover
    from .music_agent import MusicAgent

# Findings a generative attempt raises that ``Guidance`` can actually act on
# (``agent/guidance.py``'s ``guidance_from_lessons``); a failure of this kind
# is worth a guided retry rather than a level drop.
GUIDANCE_ACTIONABLE = {
    "outside_swara", "forbidden_swara", "wrong_direction", "no_cadence",
    "too_many_leaps", "not_original", "no_idiom", "repetitive", "no_gamaka",
    "neighbour_drift",
}

# A listening/recall exercise family that failed suggests trying a sibling
# family next, not the same one again.
SIBLING_EXERCISE = {
    "swara": "variant", "variant": "swara",
    "tonic": "interval", "interval": "tonic",
}


class RagaTrainer:
    """Wraps a :class:`MusicAgent` and a ``FactoryStore`` as a Trainer."""

    def __init__(self, agent: "MusicAgent", store) -> None:
        self.agent = agent
        self.store = store
        self._beats: Dict[str, int] = {}
        self._harder_wanted: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _raaga_and_unit(self, lesson: Lesson) -> Tuple[str, Optional[Unit]]:
        unit = self.agent.curriculum.unit(lesson.origin) if lesson.origin else None
        if unit is not None:
            return unit.raaga_name or self.agent.curriculum.current_raaga(), unit
        return self.agent.curriculum.current_raaga(), None

    def _fact_keys(self, unit: Optional[Unit], raaga: str) -> List[str]:
        if unit is not None and unit.params.get("facts"):
            return list(unit.params["facts"])
        return [k for k in CORE_KEYS if self.agent.repo.best_fact(raaga, k)]

    def _is_valid_phrase(self, raaga, text: str) -> bool:
        tokens = (text or "").split()
        if not tokens:
            return False
        allowed = set(raaga.allowed)
        bases = [parse_swara(t)[0] for t in tokens]
        if any(b not in allowed for b in bases):
            return False
        ascending_ok = set(raaga.ascending)
        descending_ok = set(raaga.descending)
        midi = [raaga.midi(t, TONIC) for t in tokens]
        for i in range(1, len(tokens)):
            if midi[i] > midi[i - 1] and bases[i] not in ascending_ok:
                return False
            if midi[i] < midi[i - 1] and bases[i] not in descending_ok:
                return False
        return True

    def _second_raaga(self, raaga_name: str) -> str:
        """The next raaga sharing a melakarta, or the first other raaga in
        the library (T8, generalisation)."""
        raaga_obj = self.agent.library.get(raaga_name)
        all_raagas = self.agent.library.all()
        if raaga_obj is not None and raaga_obj.melakarta is not None:
            same = sorted((r for r in all_raagas
                          if r.melakarta == raaga_obj.melakarta
                          and r.name != raaga_name), key=lambda r: r.name)
            if same:
                return same[0].name
        others = sorted(r.name for r in all_raagas if r.name != raaga_name)
        return others[0] if others else raaga_name

    # ------------------------------------------------------------------
    # next_lesson
    # ------------------------------------------------------------------
    def _unexamined_stated(self, profile: AgentProfile) -> Optional[Lesson]:
        """A lesson from a studied source that has never been examined on.

        The curriculum is the spine and stays so: a stated lesson jumps the
        queue exactly once, when it is new, so that what a source taught is
        tested while the creator still remembers approving it - and then the
        curriculum carries on.  A concept already examined comes round again
        through the ladder like anything else.
        """
        for lesson in self.store.lessons(domain="carnatic-music"):
            if not is_stated(lesson):
                continue
            record = self.store.mastery(profile.id, lesson.concept)
            if record is None or not record.evidence:
                return lesson
        return None

    def next_lesson(self, profile: AgentProfile,
                    history: Sequence[TestResult]) -> Optional[Lesson]:
        raaga = self.agent.curriculum.current_raaga()
        stated = self._unexamined_stated(profile)
        if stated is not None:
            return stated
        unit = self.agent.curriculum.next_unit(raaga)
        if unit is None:
            return None
        concept = (f"{unit.curriculum_unit_id}:{unit.raaga_name}"
                  if unit.raaga_name else unit.curriculum_unit_id)
        cached = self.store.lessons(domain="carnatic-music", concept=concept)
        if cached:
            return cached[0]
        lesson = self._build_lesson(unit, concept)
        self.store.save_lesson(lesson)
        return lesson

    def _build_lesson(self, unit: Unit, concept: str) -> Lesson:
        raaga_name = unit.raaga_name or self.agent.curriculum.current_raaga()
        raaga_obj = self.agent.library.get(raaga_name)
        fact_keys = self._fact_keys(unit, raaga_name)

        source_titles: List[str] = []
        for key in fact_keys:
            fact = self.agent.repo.best_fact(raaga_name, key)
            if fact is not None:
                source = self.agent.repo.source(fact.source_id)
                if source is not None and source.title not in source_titles:
                    source_titles.append(source.title)
        for phrase in self.agent.repo.phrases(raaga=raaga_name, limit=5):
            source = self.agent.repo.source(phrase.source_id)
            if source is not None and source.title not in source_titles:
                source_titles.append(source.title)

        explanation = "; ".join(MEANINGS[k] for k in fact_keys if k in MEANINGS) \
            or unit.learning_goal

        examples = [" ".join(p.swaras) for p in
                   self.agent.repo.phrases(raaga=raaga_name, limit=3)]
        if not examples and raaga_obj is not None and raaga_obj.prayogas:
            examples = [" ".join(p) for p in raaga_obj.prayogas[:3]]

        counterexamples: List[str] = []
        if raaga_obj is not None:
            rng = random.Random(practice_seed(unit.id, salt="counterexample"))
            for _ in range(2):
                counterexamples.append(
                    " ".join(self.agent.practice._corrupt(raaga_obj, rng)))

        unit_lessons = self.agent.repo.lessons(raaga=raaga_name, unit_id=unit.id,
                                               limit=10)
        common_errors = [f"{l.kind}: {l.failure_reason}" for l in unit_lessons]
        remediation = sorted({l.kind for l in unit_lessons})

        library_facts = all(_reference_value(raaga_obj, k) for k in fact_keys) \
            if raaga_obj is not None and fact_keys else False
        knowledge_class = (KnowledgeClass.HARD
                           if unit.skill_type == "recall.fact" and library_facts
                           else KnowledgeClass.HEURISTIC)

        return Lesson(
            domain="carnatic-music", concept=concept, objective=unit.learning_goal,
            prerequisites=unit.prerequisites(), source_knowledge=source_titles,
            explanation=explanation, examples=examples,
            counterexamples=counterexamples, practice_tasks=[unit.skill_type],
            test_tasks=[unit.skill_type],
            expected_behavior=f"score at least {unit.minimum_pass_score:.2f} to pass",
            common_errors=common_errors, remediation=remediation,
            knowledge_class=knowledge_class, origin=unit.id)

    # ------------------------------------------------------------------
    # check_reiteration
    # ------------------------------------------------------------------
    def check_reiteration(self, lesson: Lesson,
                          reiteration: Reiteration) -> ReiterationCheck:
        check = ReiterationCheck()
        raaga_name, unit = self._raaga_and_unit(lesson)
        raaga_obj = self.agent.library.get(raaga_name)
        fact_keys = self._fact_keys(unit, raaga_name)

        restated = _base_tokens(reiteration.restate)
        if fact_keys and raaga_obj is not None:
            check.restate_ok = all(
                not _base_tokens(_reference_value(raaga_obj, k))
                or _base_tokens(_reference_value(raaga_obj, k)).issubset(restated)
                for k in fact_keys)
        else:
            check.restate_ok = bool(reiteration.restate)

        explain_low = (reiteration.explain or "").lower()
        meaning_keys = [k for k in fact_keys if k in MEANINGS]
        if meaning_keys:
            check.explain_ok = all(
                any(word in explain_low for word in MEANINGS[k].split())
                for k in meaning_keys)
        else:
            check.explain_ok = bool(reiteration.explain)

        prereqs = unit.prerequisites() if unit is not None else []
        check.connect_ok = all(
            self.agent.curriculum.repo.progress(p).status == "passed"
            for p in prereqs)

        check.example_ok = bool(raaga_obj) and self._is_valid_phrase(
            raaga_obj, reiteration.example)
        counter_text = (reiteration.counterexample or "").split("(")[0]
        check.counterexample_ok = bool(raaga_obj) and not self._is_valid_phrase(
            raaga_obj, counter_text)

        if not check.restate_ok:
            check.notes.append("restatement does not match the library")
        if not check.explain_ok:
            check.notes.append("explanation misses the concept's meaning")
        if not check.connect_ok:
            check.notes.append("a named prerequisite is not actually passed")
        if not check.example_ok:
            check.notes.append("the example is not a valid phrase")
        if not check.counterexample_ok:
            check.notes.append("the counterexample is not actually invalid")
        return check

    # ------------------------------------------------------------------
    # build_tests
    # ------------------------------------------------------------------
    def _split_for(self, n: int) -> Split:
        if n % 5 == 0:
            return Split.HIDDEN
        if n % 3 == 0:
            return Split.VALIDATION
        return Split.TRAINING

    def _novelty(self, history: Sequence[TestResult], level: TestLevel,
                seed: int, raaga_name: str) -> float:
        if not history:
            return 1.0
        seen_seeds, seen_raagas, seen_levels = set(), set(), set()
        for r in history:
            seen_levels.add(r.level)
            test = self.store.test(r.test_id)
            if test is None:
                continue
            seen_seeds.add(test.seed)
            seen_raagas.add(test.payload.get("raaga", ""))
        if seed not in seen_seeds and raaga_name not in seen_raagas:
            return 1.0
        if level not in seen_levels:
            return 0.5
        return 0.2

    def _test(self, lesson: Lesson, history: Sequence[TestResult],
             level: TestLevel, skill_type: str, params: Dict, raaga_name: str,
             attempt: int, index: int = 0, ambiguity: Optional[float] = None,
             objective: bool = True, guided: Optional[bool] = None
             ) -> TestSpec:
        split = self._split_for(attempt + index + 1)
        salt = {Split.TRAINING: "test", Split.VALIDATION: "validation",
               Split.HIDDEN: "hidden"}.get(split, "test")
        seed = practice_seed(lesson.origin or lesson.concept, attempt, salt=salt)

        params = dict(params)
        if self._harder_wanted.get(lesson.concept):
            params["length"] = int(params.get("length", 5)) + 2
            params["check_originality"] = True
            if "tolerance_semitones" in params:
                params["tolerance_semitones"] = round(
                    max(0.3, float(params["tolerance_semitones"]) * 0.6), 3)
            if "tolerance_ratio" in params:
                params["tolerance_ratio"] = round(
                    max(0.05, float(params["tolerance_ratio"]) * 0.6), 3)

        check_originality_on = bool(params.get("check_originality"))
        difficulty = round(min(1.0, level.value / 10.0
                              + (0.1 if check_originality_on else 0.0)), 3)
        if ambiguity is None:
            if skill_type.startswith("generate."):
                ambiguity = 0.4
            elif "mood" in skill_type:
                ambiguity = 0.7
            else:
                ambiguity = 0.0

        payload = {"skill_type": skill_type, "params": params,
                  "raaga": raaga_name, "origin_unit": lesson.origin}
        if guided is not None:
            payload["guided"] = guided
        # A question worded so a hard rule can settle a real disagreement
        # (docs/PLAN_agent_factory.md, "Where disputes actually arise").
        # Only set where the student's and trainer's claim are genuinely a
        # verdict about one phrase or one fact - a multi-exercise claim (the
        # exercises joined) is not a single verdict a rule can compare
        # against, so these skill types run exactly one exercise here.
        if skill_type in ("classify.valid", "correct.phrase"):
            params["exercises"] = 1
            payload["question"] = f"is the phrase valid in {raaga_name}?"
        elif skill_type.startswith("generate."):
            # Both sides state a grammar verdict on what was generated (see
            # grade() and RagaStudent._generation_claim), so the same
            # question fits and a hard rule can settle it.
            payload["question"] = f"is the phrase valid in {raaga_name}?"
        elif skill_type in ("explain", "recall.fact"):
            keys = list(params.get("facts", []))
            if len(keys) == 1:
                payload["question"] = f"what is the {keys[0]} of {raaga_name}"

        novelty = self._novelty(history, level, seed, raaga_name)
        return TestSpec(
            capability=lesson.concept, level=level, novelty=novelty,
            difficulty=difficulty, ambiguity=ambiguity, objective=objective,
            split=split, seed=seed, lesson_id=lesson.id, payload=payload)

    def _skill_test(self, lesson: Lesson, unit: Optional[Unit],
                    history: Sequence[TestResult], level: TestLevel,
                    raaga_name: str, attempt: int, guided: bool,
                    index: int = 0) -> TestSpec:
        if unit is None:
            return self._test(lesson, history, level, "generate.pattern",
                              {"length": 5}, raaga_name, attempt, index,
                              guided=guided)
        return self._test(lesson, history, level, unit.skill_type,
                          dict(unit.params), raaga_name, attempt, index,
                          guided=guided)

    def build_tests(self, lesson: Lesson, profile: AgentProfile,
                    history: Sequence[TestResult]) -> List[TestSpec]:
        from ..factory.mastery import next_test_level

        raaga_name, unit = self._raaga_and_unit(lesson)
        record = self.store.mastery(profile.id, lesson.concept)
        level = next_test_level(record)
        attempt = sum(1 for r in history if r.lesson_id == lesson.id)
        if is_stated(lesson):
            # A transcript can be examined on to the point of explaining it
            # and no further, so the ladder stops there rather than asking
            # for an application nobody could grade honestly.
            level = min(level, TestLevel.T2_EXPLANATION)
            tests = self._stated_tests(level, lesson, raaga_name, history,
                                       attempt)
            if level < TestLevel.T2_EXPLANATION:
                tests.extend(self._stated_tests(
                    TestLevel(int(level) + 1), lesson, raaga_name, history,
                    attempt, offset=len(tests)))
            self._harder_wanted.pop(lesson.concept, None)
            return tests[:4]
        # ``_test`` reads ``self._harder_wanted`` for every test built below;
        # consumed once, after the whole batch, so a harder variant is asked
        # for exactly by the next ``build_tests`` call and no other.

        # Tests at the level the student has earned, and one rung above it:
        # the adaptive trainer promotes to the higher rung only after a
        # calibrated streak, and it can only promote to a test that exists.
        tests: List[TestSpec] = self._tests_at(level, lesson, unit, raaga_name,
                                               history, attempt)
        if level < TestLevel.T10_REAL_WORLD:
            # T0 and T1 are one rung (recognition and recall are built
            # together), so the rung above either is explanation.
            higher = (TestLevel.T2_EXPLANATION if level <= TestLevel.T1_RECALL
                      else TestLevel(int(level) + 1))
            tests.extend(self._tests_at(higher, lesson, unit, raaga_name,
                                        history, attempt, offset=len(tests)))
        while len(tests) < 2:
            tests.append(self._test(
                lesson, history, level, "recall.fact",
                {"facts": self._fact_keys(unit, raaga_name)}, raaga_name,
                attempt, len(tests)))
        self._harder_wanted.pop(lesson.concept, None)
        return tests[:4]

    def _stated_tests(self, level: TestLevel, lesson: Lesson, raaga_name: str,
                      history: Sequence[TestResult], attempt: int,
                      offset: int = 0) -> List[TestSpec]:
        """The rung for a lesson built from what somebody *said*.

        A stated lesson has no curriculum unit behind it and no library entry
        to be graded against: its authority is the source.  So the test asks
        what the agent retained from studying it, and the grader compares
        that with what the source actually taught.  The agent answers from the
        knowledge base rather than from the lesson, or it would be reading
        the answer back.

        Recognition, recall and explanation only.  An application test built
        from a transcript would grade the agent on something nobody heard.
        """
        payload = {"concept": lesson.concept,
                   "expects": list(lesson.examples)[:6],
                   "explanation": lesson.explanation[:600]}
        if level == TestLevel.T0_RECOGNITION:
            return [self._test(lesson, history, TestLevel.T0_RECOGNITION,
                               "recognise.stated", payload, raaga_name,
                               attempt, offset)]
        if level == TestLevel.T1_RECALL:
            return [self._test(lesson, history, TestLevel.T1_RECALL,
                               "recall.stated", payload, raaga_name,
                               attempt, offset)]
        return [self._test(lesson, history, TestLevel.T2_EXPLANATION,
                           "explain.stated", payload, raaga_name,
                           attempt, offset, objective=False)]

    def _tests_at(self, level: TestLevel, lesson: Lesson, unit: Optional[Unit],
                  raaga_name: str, history: Sequence[TestResult], attempt: int,
                  offset: int = 0) -> List[TestSpec]:
        """The ladder rung as exercises the practice engine can run."""
        if is_stated(lesson):
            return self._stated_tests(level, lesson, raaga_name, history,
                                      attempt, offset)
        tests: List[TestSpec] = []
        if level in (TestLevel.T0_RECOGNITION, TestLevel.T1_RECALL):
            tests.append(self._test(lesson, history, TestLevel.T0_RECOGNITION,
                                    "listen.identify", {"identify": "swara"},
                                    raaga_name, attempt, 0))
            tests.append(self._test(lesson, history, TestLevel.T1_RECALL,
                                    "recall.fact",
                                    {"facts": self._fact_keys(unit, raaga_name)},
                                    raaga_name, attempt, 1))
        elif level == TestLevel.T2_EXPLANATION:
            tests.append(self._test(lesson, history, TestLevel.T2_EXPLANATION,
                                    "explain",
                                    {"facts": self._fact_keys(unit, raaga_name)},
                                    raaga_name, attempt, 0, objective=False))
        elif level == TestLevel.T3_CONTROLLED_APPLICATION:
            tests.append(self._skill_test(
                lesson, unit, history, TestLevel.T3_CONTROLLED_APPLICATION,
                raaga_name, attempt, guided=True))
        elif level == TestLevel.T4_INDEPENDENT_APPLICATION:
            tests.append(self._skill_test(
                lesson, unit, history, TestLevel.T4_INDEPENDENT_APPLICATION,
                raaga_name, attempt, guided=False))
        elif level == TestLevel.T5_VARIATION:
            # A changed example of the unit's own skill: variations for a
            # generative unit, tighter tolerances for a listening one.
            if unit is not None and unit.skill_type != "generate.pattern":
                params = dict(unit.params)
                for key in ("tolerance_semitones", "tolerance_ratio"):
                    if key in params:
                        params[key] = round(float(params[key]) * 0.7, 3)
                params["variant"] = True
                tests.append(self._test(
                    lesson, history, TestLevel.T5_VARIATION, unit.skill_type,
                    params, raaga_name, attempt, 0, guided=False))
            else:
                params = dict(unit.params) if unit is not None else {"length": 5}
                params.update({"variations": 3, "use_learned_phrases": True})
                params.setdefault("length", 5)
                tests.append(self._test(
                    lesson, history, TestLevel.T5_VARIATION, "generate.pattern",
                    params, raaga_name, attempt, 0))
        elif level == TestLevel.T6_ERROR_DETECTION:
            tests.append(self._test(
                lesson, history, TestLevel.T6_ERROR_DETECTION, "classify.valid",
                {"mode": "in_raaga_vs_out"}, raaga_name, attempt, 0))
        elif level == TestLevel.T7_CORRECTION:
            tests.append(self._test(
                lesson, history, TestLevel.T7_CORRECTION, "correct.phrase",
                {}, raaga_name, attempt, 0))
        elif level == TestLevel.T8_GENERALIZATION:
            second = self._second_raaga(raaga_name)
            tests.append(self._skill_test(
                lesson, unit, history, TestLevel.T8_GENERALIZATION,
                second, attempt, guided=False))
        elif level == TestLevel.T9_ADVERSARIAL:
            tests.append(self._test(
                lesson, history, TestLevel.T9_ADVERSARIAL, "classify.valid",
                {"mode": "neighbour_drift"}, raaga_name, attempt, 0))
        else:  # T10
            tests.append(self._test(
                lesson, history, TestLevel.T10_REAL_WORLD, "generate.section",
                {"section": "pallavi", "seconds": 20}, raaga_name, attempt, 0,
                objective=False))
        # Index the batch so splits and seeds differ from the lower rung's.
        if offset:
            for i, test in enumerate(tests):
                split = self._split_for(attempt + offset + i + 1)
                test.split = split
                salt = {Split.TRAINING: "test", Split.VALIDATION: "validation",
                        Split.HIDDEN: "hidden"}.get(split, "test")
                test.seed = practice_seed(lesson.origin or lesson.concept,
                                          attempt + offset + i, salt=salt)
        return tests

    # ------------------------------------------------------------------
    # grade
    # ------------------------------------------------------------------
    def grade(self, test: TestSpec, performance) -> TestResult:
        report = performance.payload.get("report") if performance.payload else None
        threshold = 0.7 + (0.05 if test.level.value >= TestLevel.T5_VARIATION.value
                           else 0.0)
        raaga_name = test.payload.get("raaga", "")
        raaga_obj = self.agent.library.get(raaga_name)

        skill = str(test.payload.get("skill_type", ""))
        if report is None and skill.endswith(".stated"):
            return self._grade_stated(test, performance, threshold)
        if report is None and skill == "explain":
            return self._grade_explain(test, performance, raaga_obj, threshold)

        score = report.score if report is not None else 0.0
        passed = score >= threshold

        trainer_claim = "invalid"
        trainer_confidence = 0.8 if test.objective else 0.5
        evidence: List[str] = []
        failure_mode = ""

        if report is not None:
            if report.evaluation is not None:
                trainer_confidence = report.evaluation.confidence
                if raaga_obj is not None and report.artifacts:
                    # The trainer's claim is a grammar verdict on what was
                    # generated, judged against the LIBRARY (hard knowledge):
                    # the student states the same kind of verdict from its
                    # learned view, so a disagreement is about one phrase
                    # and a hard rule can settle it.  The score stays the
                    # report's; the claim is not the score.
                    contested = None
                    for notes in report.artifacts:
                        tokens = [n.swara for n in notes]
                        if not self.agent.practice._judge_valid(raaga_obj, tokens):
                            contested = tokens
                            break
                    trainer_claim = "invalid" if contested else "valid"
                    shown = contested or [n.swara for n in report.artifacts[0]]
                    evidence.append("phrase: " + " ".join(shown))
                    evidence.append(f"raaga: {raaga_name}")
                evidence.extend(f.text for f in report.findings)
                if report.findings:
                    failure_mode = report.findings[0].kind
            else:
                # An objective exercise: the trainer's claim is what it
                # expected, in the same form as the student's answers, so
                # the two are comparable and a wrong answer is visibly a
                # wrong answer rather than a dispute.
                failed = [e for e in report.exercises if not e.passed]
                trainer_claim = "; ".join(e.expected for e in report.exercises
                                          if e.expected) or (
                    "valid" if not failed else "invalid")
                evidence = [e.detail for e in report.exercises if e.detail]
                if failed:
                    failure_mode = _exercise_family(failed[0].name)

        return TestResult(
            test_id=test.id, agent_id=self.agent.profile().id,
            lesson_id=test.lesson_id, level=test.level, split=test.split,
            score=score, passed=passed, student_claim=performance.claim,
            student_confidence=performance.confidence,
            trainer_claim=trainer_claim, trainer_confidence=trainer_confidence,
            failure_mode=failure_mode, evidence=evidence,
            duration_seconds=performance.duration_seconds)

    def _grade_stated(self, test: TestSpec, performance,
                      threshold: float) -> TestResult:
        """Did the agent retain what the source taught?

        The reference is the lesson, not the library: a stated lesson's
        authority is the source it came from, and the agent's answer comes
        from the knowledge base, so this compares what was kept against what
        was said.  Word overlap rather than anything cleverer, because the
        judge has to be deterministic and offline like every other one here;
        a provider-backed reading is what the escalation hook is for.
        """
        params = test.payload.get("params", {})
        expected = [str(e) for e in params.get("expects", []) if str(e).strip()]
        reference = " ".join(expected) or str(params.get("explanation", ""))
        said = _base_tokens(reference)
        got = _base_tokens(performance.output)
        if not said:
            score = 1.0 if performance.output.strip() else 0.0
        else:
            score = round(len(said & got) / len(said), 3)
        passed = score >= threshold
        return TestResult(
            test_id=test.id, agent_id=self.agent.profile().id,
            lesson_id=test.lesson_id, level=test.level, split=test.split,
            score=score, passed=passed, student_claim=performance.claim,
            student_confidence=performance.confidence,
            trainer_claim="valid" if passed else "invalid",
            trainer_confidence=0.6,
            failure_mode="" if passed else "stated_not_retained",
            evidence=expected[:3],
            duration_seconds=performance.duration_seconds)

    def _grade_explain(self, test: TestSpec, performance, raaga_obj,
                       threshold: float) -> TestResult:
        keys = test.payload.get("params", {}).get("facts", [])
        hits = 0
        for key in keys:
            expected = _reference_value(raaga_obj, key) if raaga_obj else ""
            if expected and _base_tokens(expected).issubset(
                    _base_tokens(performance.output)):
                hits += 1
        score = round(hits / len(keys), 3) if keys else \
            (1.0 if performance.output else 0.0)
        passed = score >= threshold
        return TestResult(
            test_id=test.id, agent_id=self.agent.profile().id,
            lesson_id=test.lesson_id, level=test.level, split=test.split,
            score=score, passed=passed, student_claim=performance.claim,
            student_confidence=performance.confidence,
            trainer_claim="valid" if passed else "invalid",
            trainer_confidence=0.8 if test.objective else 0.5,
            failure_mode="" if passed else "explain_incomplete",
            evidence=[f"{k}: {_reference_value(raaga_obj, k)}" for k in keys]
            if raaga_obj else [],
            duration_seconds=performance.duration_seconds)

    # ------------------------------------------------------------------
    # remediate
    # ------------------------------------------------------------------
    def remediate(self, profile: AgentProfile, lesson: Lesson,
                 failures: Sequence[TestResult]) -> Remediation:
        kinds = [f.failure_mode for f in failures if f.failure_mode]
        shared = [k for k in set(kinds) if kinds.count(k) >= 2]
        actionable = sorted(k for k in shared if k in GUIDANCE_ACTIONABLE)
        if actionable:
            return Remediation(
                kind="guided", lesson_id=lesson.id,
                detail=", ".join(actionable),
                next_level=TestLevel.T3_CONTROLLED_APPLICATION,
                payload={"guided": True})

        listening = [f for f in failures
                    if f.level in (TestLevel.T0_RECOGNITION, TestLevel.T1_RECALL)]
        if listening:
            mode = ""
            for f in failures:
                test = self.store.test(f.test_id)
                if test is not None:
                    mode = test.payload.get("params", {}).get("identify", "")
                    if mode:
                        break
            sibling = SIBLING_EXERCISE.get(mode, "variant")
            return Remediation(
                kind="different_practice", lesson_id=lesson.id,
                detail=f"{mode or 'listening'} -> {sibling}",
                payload={"identify": sibling})

        return Remediation(
            kind="level_down", lesson_id=lesson.id,
            detail="repeated failure without a common lesson",
            next_level=self._level_down(failures))

    @staticmethod
    def _level_down(failures: Sequence[TestResult]) -> TestLevel:
        if not failures:
            return TestLevel.T0_RECOGNITION
        lowest = min(f.level for f in failures)
        return TestLevel(max(0, lowest.value - 1))

    # ------------------------------------------------------------------
    # learn_from
    # ------------------------------------------------------------------
    def learn_from(self, result: TestResult) -> None:
        if not result.passed:
            return
        beats = self._beats.get(result.test_id, 0) + 1
        self._beats[result.test_id] = beats
        if beats >= 3:
            self.store.retire_test(result.test_id)
            concept = None
            test = self.store.test(result.test_id)
            if test is not None:
                concept = test.capability
            if concept:
                self._harder_wanted[concept] = True


def _exercise_family(exercise_name: str) -> str:
    """"name the swara 3" -> "swara": the exercise family a listening/recall
    failure belongs to, for remediation's sibling-exercise lookup."""
    base = exercise_name.rsplit(" ", 1)[0] if exercise_name[-1:].isdigit() \
        else exercise_name
    for family in ("swara", "variant", "tonic", "interval", "tempo"):
        if family in base:
            return family
    return base.strip()
