"""RagaStudent: the Student role (document 02) over a ``MusicAgent``.

Nothing here calls a model.  Reiteration is built from what the knowledge
base already holds; a test is performed by mapping its ladder level onto the
existing practice engine; a correction is written back as a lesson.
"""
from __future__ import annotations

import random
import re
import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from ..factory.models import (AgentProfile, Lesson, Performance, Reiteration,
                              TestSpec)
from ..raaga.library import parse_swara
from .curriculum import Unit
from .guidance import build_guidance
from .knowledge import Fact
from .knowledge import Lesson as KnowledgeLesson
from .learned import CORE_KEYS, knowledge_confidence
from .practice import TONIC, practice_seed

if TYPE_CHECKING:  # pragma: no cover
    from .music_agent import MusicAgent

# Document 04's R3: what a concept is *for*, in the agent's own words.  Shared
# with RagaTrainer.check_reiteration, which looks for these same key words.
MEANINGS = {
    "jeeva": "jeeva carry the raaga's identity",
    "nyasa": "nyasa are resting notes phrases close on",
    "graha": "graha are starting notes",
    "arohanam": "arohanam is the permitted way up",
    "avarohanam": "avarohanam the permitted way down",
    "prayogas": "prayogas are its characteristic phrases",
    "gamaka": "gamaka its ornaments",
}

# A ruling's correction text that names a fact, written in the shape
# FactRule.decide produces: "<raaga>'s <key> is <value> by the library ...".
def _looks_like_swaras(text: str) -> bool:
    """Every token is a swara name, so a rule can judge it as a phrase."""
    from ..raaga.library import SWARA_SEMITONES, parse_swara
    tokens = (text or "").split()
    return bool(tokens) and all(parse_swara(t)[0] in SWARA_SEMITONES
                                for t in tokens)


FACT_CORRECTION = re.compile(
    r"^(?P<raaga>.+?)'s (?P<key>\w+) is (?P<value>.+?) by the library", re.I)


class RagaStudent:
    """Wraps a :class:`MusicAgent` as a framework ``Student``."""

    def __init__(self, agent: "MusicAgent", profile: AgentProfile) -> None:
        self.agent = agent
        self._profile = profile

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    # ------------------------------------------------------------------
    # helpers shared by every step
    # ------------------------------------------------------------------
    def _unit(self, lesson: Lesson) -> Optional[Unit]:
        return self.agent.curriculum.unit(lesson.origin) if lesson.origin else None

    def _raaga_name(self, lesson: Lesson) -> str:
        unit = self._unit(lesson)
        if unit is not None and unit.raaga_name:
            return unit.raaga_name
        return self.agent.curriculum.current_raaga()

    def _fact_keys(self, unit: Optional[Unit], raaga: str) -> List[str]:
        if unit is not None and unit.params.get("facts"):
            return list(unit.params["facts"])
        return [k for k in CORE_KEYS if self.agent.repo.best_fact(raaga, k)]

    # ------------------------------------------------------------------
    # step 1: acquire
    # ------------------------------------------------------------------
    def acquire(self, lesson: Lesson) -> None:
        unit = self._unit(lesson)
        if unit is None:
            return
        raaga = unit.raaga_name or self.agent.curriculum.current_raaga()
        if not self.agent.repo.facts(raaga):
            self.agent.research.seed_structural_knowledge(raaga)
        if unit.retry_policy == "ingest_more_sources":
            limit = int(getattr(self.agent.settings,
                                "learning_max_sources_per_lesson", 4))
            candidates = self.agent.research.find_sources(
                raaga, unit.learning_goal, limit)
            for candidate in candidates[:limit]:
                self.agent.research.ingest(candidate)

    # ------------------------------------------------------------------
    # steps 2-5: reiterate
    # ------------------------------------------------------------------
    def _restate(self, raaga: str, keys: Sequence[str]) -> str:
        sentences = []
        for key in keys:
            fact = self.agent.repo.best_fact(raaga, key)
            if fact:
                sentences.append(f"The {key} of {raaga} is {fact.value}.")
        return " ".join(sentences) if sentences else \
            f"I have not learned {raaga}'s facts for this concept yet."

    def _explain(self, raaga: str, keys: Sequence[str]) -> str:
        parts = []
        for key in keys:
            meaning = MEANINGS.get(key)
            if not meaning:
                continue
            fact = self.agent.repo.best_fact(raaga, key)
            if fact is not None:
                source = self.agent.repo.source(fact.source_id)
                origin = source.title if source else "my own practice"
                parts.append(f"{meaning} (from {origin}, confidence "
                             f"{fact.confidence:.2f})")
            else:
                parts.append(meaning)
        return "; ".join(parts) if parts else "nothing to explain yet"

    def _connect(self, lesson: Lesson, unit: Optional[Unit]) -> str:
        if unit is None:
            return "no prerequisites recorded"
        prereqs = unit.prerequisites()
        passed = [p for p in prereqs
                 if self.agent.curriculum.repo.progress(p).status == "passed"]
        parts = []
        if passed:
            parts.append(f"builds on {', '.join(passed)}")
        related = [t for t in lesson.source_knowledge[:3] if t]
        if related:
            parts.append(f"related facts from {', '.join(related)}")
        return "; ".join(parts) if parts else "no prerequisites recorded"

    def _example(self, raaga: str) -> str:
        phrases = self.agent.repo.phrases(raaga=raaga, limit=1)
        if phrases:
            return " ".join(phrases[0].swaras)
        fact = self.agent.repo.best_fact(raaga, "arohanam")
        return fact.value if fact else ""

    def _counterexample(self, raaga: str) -> str:
        raaga_obj = self.agent.library.get(raaga)
        if raaga_obj is None:
            return ""
        rng = random.Random(practice_seed(f"reiterate:{raaga}", 0))
        tokens = self.agent.practice._corrupt(raaga_obj, rng)
        allowed = set(raaga_obj.allowed)
        outside = [parse_swara(t)[0] for t in tokens
                  if parse_swara(t)[0] not in allowed]
        reason = (f"uses {outside[0]}, which {raaga} does not have" if outside
                  else f"does not stay inside {raaga}")
        return f"{' '.join(tokens)} ({reason})"

    def reiterate(self, lesson: Lesson) -> Reiteration:
        unit = self._unit(lesson)
        raaga = self._raaga_name(lesson)
        keys = self._fact_keys(unit, raaga)

        report = self.agent.practice.run(unit, raaga) if unit is not None else None
        apply_summary = report.summary() if report is not None else \
            "nothing to apply yet"
        apply_score = report.score if report is not None else 0.0

        self_check_parts = []
        if report is not None and report.evaluation is not None:
            weakest = [name for name, _ in report.evaluation.weakest(2)]
            if weakest:
                self_check_parts.append(f"weakest: {', '.join(weakest)}")
        open_lessons = (self.agent.repo.lessons(unit_id=unit.id)
                        if unit is not None else [])
        self_check_parts.append(f"{len(open_lessons)} open lesson(s) for this unit")

        return Reiteration(
            lesson_id=lesson.id, agent_id=self.profile.id,
            restate=self._restate(raaga, keys),
            explain=self._explain(raaga, keys),
            connect=self._connect(lesson, unit),
            example=self._example(raaga),
            counterexample=self._counterexample(raaga),
            apply_summary=apply_summary, apply_score=apply_score,
            self_check="; ".join(self_check_parts),
            retest_due_at=time.time() + self.agent.curriculum.revisit_after)

    # ------------------------------------------------------------------
    # step 6: perform a test
    # ------------------------------------------------------------------
    def perform(self, test: TestSpec) -> Performance:
        payload = dict(test.payload or {})
        raaga = payload.get("raaga") or self.agent.curriculum.current_raaga()
        skill_type = payload.get("skill_type", "generate.pattern")
        params = dict(payload.get("params", {}))

        if skill_type == "explain":
            return self._perform_explain(raaga, params)

        unit = Unit(
            curriculum_unit_id=f"test:{test.id}", skill_type=skill_type,
            raaga_name=raaga, learning_goal=test.capability,
            exercises=int(params.pop("exercises", 3)),
            minimum_pass_score=float(params.pop("minimum_pass_score", 0.7)),
            params=params)
        guidance = None
        if payload.get("guided"):
            guidance = build_guidance(self.agent.repo, raaga,
                                      payload.get("origin_unit", ""))

        start = time.time()
        report = self.agent.practice.run(unit, raaga, seed=test.seed,
                                         guidance=guidance)
        duration = time.time() - start

        objective_family = skill_type.startswith(
            ("classify.", "recall.", "listen.", "correct."))
        if objective_family:
            claim = "; ".join(e.heard for e in report.exercises if e.heard)
        else:
            claim = self._generation_claim(raaga, report, unit)

        confidence = (report.evaluation.confidence if report.evaluation is not None
                     else knowledge_confidence(self.agent.repo, raaga)["overall"])

        evidence: List[str] = []
        for notes in report.artifacts:
            evidence.append("phrase: " + " ".join(n.swara for n in notes))
        if not report.artifacts:
            # The phrase under judgement, not the student's answer about
            # it: a classification exercise keeps its tokens in ``detail``
            # and its verdict in ``heard``; a repair keeps the repaired
            # line in ``heard``.
            for exercise in report.exercises:
                phrase = (exercise.detail if skill_type == "classify.valid"
                          else exercise.heard)
                if phrase and _looks_like_swaras(phrase):
                    evidence.append("phrase: " + phrase)
        evidence.append(f"raaga: {raaga}")
        evidence.append(f"score: {report.score}")
        evidence.extend(f"{e.name}: {e.detail}" for e in report.exercises
                        if e.detail)

        return Performance(output=report.summary(), claim=claim,
                           confidence=confidence, evidence=evidence,
                           duration_seconds=duration, payload={"report": report})

    def _generation_claim(self, raaga: str, report, unit: Unit) -> str:
        """The student's own judgement of a generative attempt, from the
        LEARNED view of the raaga - not the library."""
        learned_raaga_obj, _ = self.agent.raaga_for_composition(raaga)
        if learned_raaga_obj is None or not report.artifacts:
            return "invalid"
        # A grammar verdict from what the student believes the raaga to be:
        # every generated line must use only its swaras and move the way
        # its arohanam and avarohanam allow.  The trainer states the same
        # verdict from the library, so the two differ exactly when what was
        # learned contradicts what is known - the case the Judge is for.
        for notes in report.artifacts:
            tokens = [n.swara for n in notes]
            if not self.agent.practice._judge_valid(learned_raaga_obj, tokens):
                return "invalid"
        return "valid"

    def _perform_explain(self, raaga: str, params: dict) -> Performance:
        keys = list(params.get("facts", [])) or self._fact_keys(None, raaga)
        output = self._explain(raaga, keys)
        confidence = knowledge_confidence(self.agent.repo, raaga)["overall"]
        evidence = [f"raaga: {raaga}"]
        for key in keys:
            fact = self.agent.repo.best_fact(raaga, key)
            if fact is not None:
                evidence.append(f"{key}: {fact.value}")
        return Performance(output=output, claim=output, confidence=confidence,
                           evidence=evidence, payload={})

    # ------------------------------------------------------------------
    # correction
    # ------------------------------------------------------------------
    def apply_correction(self, correction: str, lesson: Optional[Lesson]) -> None:
        raaga = self._raaga_name(lesson) if lesson is not None else \
            self.agent.curriculum.current_raaga()
        unit = self._unit(lesson) if lesson is not None else None
        stored = KnowledgeLesson(
            raaga=raaga, unit_id=unit.id if unit is not None else "",
            task=lesson.objective if lesson is not None else "",
            kind="judge_correction", dimension="dispute",
            failure_reason=correction, correction=correction,
            source_run="judge", confidence=0.9)
        self.agent.repo.add_lesson(stored)

        match = FACT_CORRECTION.match(correction or "")
        if match:
            # A ruling from hard knowledge is as sure as the library: the
            # right value goes in at full confidence and every claim that
            # contradicted it is overruled, so the learned view changes.
            fact_raaga = match.group("raaga").strip()
            key = match.group("key").strip()
            value = match.group("value").strip()
            self.agent.repo.add_fact(Fact(
                raaga=fact_raaga, key=key, value=value, confidence=1.0,
                notes="corrected by a Judge ruling"))
            self.agent.repo.overrule_facts(
                fact_raaga, key, value, note="overruled by a Judge ruling")
            if key == "swaras":
                # The scale claims that put a foreign note into the
                # inventory are contradicted by the same ruling.
                from ..raaga.library import parse_swara
                ruled = {parse_swara(t)[0] for t in value.split()}
                for scale_key in ("arohanam", "avarohanam"):
                    for held in self.agent.repo.facts(fact_raaga, scale_key):
                        bases = {parse_swara(t)[0] for t in held.value.split()}
                        if bases - ruled:
                            self.agent.repo.overrule_fact(
                                fact_raaga, scale_key, held.value,
                                note="overruled by a Judge ruling on the swaras")
