"""A domain-free proof of the Agent Factory core: English plural rules.

Not a product - a test fixture, kept deliberately small.  ``ToyStudent`` and
``ToyTrainer`` implement the ``Student``/``Trainer`` protocols from
``raagacomposer.factory.protocols`` without knowing anything about music,
so the acceptance tests in this directory can exercise the real core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from raagacomposer.factory.models import (AgentProfile, Dispute, KnowledgeClass,
                                          Lesson, Performance, Reiteration,
                                          ReiterationCheck, Remediation,
                                          Ruling, Split, TestLevel, TestResult,
                                          TestSpec, new_id)

# The hard rule table: a suffix -> the plural rule that applies to words
# ending in it.  "default" is the ordinary "add s" rule.
RULES = {
    "default": "add s",
    "s": "after s, x, ch, sh add es",
    "x": "after s, x, ch, sh add es",
    "ch": "after s, x, ch, sh add es",
    "sh": "after s, x, ch, sh add es",
}
HEURISTIC = "words ending in y usually take ies"


def _matched_suffix(word: str) -> str:
    for suffix in ("ch", "sh", "s", "x"):
        if word.endswith(suffix):
            return suffix
    return "default"


def correct_plural(word: str) -> str:
    """The ground truth, used for grading - not for the student to peek at."""
    suffix = _matched_suffix(word)
    if suffix == "default" and word.endswith("y") and len(word) > 1 \
           and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if suffix in ("s", "x", "ch", "sh"):
        return word + "es"
    return word + "s"


def _rule_for(word: str) -> str:
    """What the hard-rule table (not the y-heuristic) says for this word."""
    suffix = _matched_suffix(word)
    return RULES[suffix]


@dataclass
class ToyStudent:
    """Holds a dict of taught rules and reiterates/answers from them only -
    it never consults ``correct_plural`` directly, so its mistakes are real."""
    agent_id: str = field(default_factory=lambda: new_id("agent"))
    name: str = "toy-student"
    domain: str = "plural-rules"
    taught_rules: Dict[str, str] = field(default_factory=dict)     # concept -> rule text
    scripted_errors: Dict[str, str] = field(default_factory=dict)  # word -> wrong answer
    seen_words: set = field(default_factory=set)
    corrections: List[str] = field(default_factory=list)
    _profile: AgentProfile = field(default_factory=lambda: AgentProfile())

    def __post_init__(self) -> None:
        self._profile.id = self.agent_id
        self._profile.name = self.name
        self._profile.domain = self.domain

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    def acquire(self, lesson: Lesson) -> None:
        self.taught_rules[lesson.concept] = lesson.explanation

    def reiterate(self, lesson: Lesson) -> Reiteration:
        rule_text = self.taught_rules.get(lesson.concept, "")
        example = lesson.examples[0] if lesson.examples else ""
        counter = lesson.counterexamples[0] if lesson.counterexamples else ""
        return Reiteration(
            lesson_id=lesson.id, agent_id=self.agent_id,
            restate=rule_text,
            explain=f"this rule matters because {lesson.objective}",
            connect=f"builds on: {', '.join(lesson.prerequisites) or 'nothing'}",
            example=example, counterexample=counter,
            apply_summary=f"applied {lesson.concept} to a practice word",
            apply_score=1.0 if rule_text else 0.0,
            self_check="I may mishandle words with unusual spellings",
            retest_due_at=0.0)

    def perform(self, test: TestSpec) -> Performance:
        word = str(test.payload.get("word", ""))
        self.seen_words.add(word)
        concept = test.capability
        taught = concept in self.taught_rules

        if word in self.scripted_errors:
            claim = self.scripted_errors[word]
            confidence = 0.8
            output = claim
        elif taught:
            claim = self._apply_taught_rule(concept, word)
            output = claim
            confidence = 0.9
        else:
            # No rule taught for this concept yet: guess with the ordinary
            # "add s" pattern and say so honestly.
            claim = word + "s"
            output = claim
            confidence = 0.4

        return Performance(output=output, claim=claim, confidence=confidence,
                           evidence=[f"used rule: {self.taught_rules.get(concept, 'guess')}"])

    def _apply_taught_rule(self, concept: str, word: str) -> str:
        rule_text = self.taught_rules.get(concept, "")
        if rule_text == HEURISTIC:
            if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
                return word[:-1] + "ies"
            return word + "s"
        if "es" in rule_text and "s, x, ch, sh" in rule_text:
            if _matched_suffix(word) in ("s", "x", "ch", "sh"):
                return word + "es"
            return word + "s"
        return word + "s"

    def apply_correction(self, correction: str, lesson: Optional[Lesson]) -> None:
        self.corrections.append(correction)
        if lesson is not None:
            self.taught_rules[lesson.concept] = correction


@dataclass
class ToyTrainer:
    """Fixed word lists per level, novelty from what this student has seen,
    grading against the hard rule table, remediation that changes the word
    list or the level."""
    domain: str = "plural-rules"
    concepts: List[str] = field(default_factory=lambda: ["add_s", "sibilant_es", "y_to_ies"])
    lesson_index: int = 0
    remediation_calls: int = 0
    remediation_kind_overrides: Dict[str, List[str]] = field(default_factory=dict)
    learned_results: List[TestResult] = field(default_factory=list)
    # Test specs are cached by (concept, level, word) so the same word/level
    # combination always maps to the same TestSpec id across calls - the
    # adaptive trainer's "unseen" and "beaten N times" tracking depends on a
    # test keeping its identity.
    _test_cache: Dict[tuple, TestSpec] = field(default_factory=dict, repr=False)
    _word_by_test_id: Dict[str, str] = field(default_factory=dict, repr=False)
    WORD_LISTS: Dict[str, Dict[TestLevel, List[str]]] = field(
        default_factory=lambda: {
        "add_s": {
            TestLevel.T0_RECOGNITION: ["cat", "dog", "book"],
            TestLevel.T1_RECALL: ["cat", "dog", "book"],
            TestLevel.T3_CONTROLLED_APPLICATION: ["cat", "dog", "pen"],
            TestLevel.T4_INDEPENDENT_APPLICATION: ["hat", "car", "lamp"],
            TestLevel.T5_VARIATION: ["desk", "chair", "phone"],
            TestLevel.T6_ERROR_DETECTION: ["cat", "dog"],
            TestLevel.T7_CORRECTION: ["cat", "dog"],
            TestLevel.T8_GENERALIZATION: ["table", "window"],
            TestLevel.T9_ADVERSARIAL: ["quiz-safe", "plan"],
            TestLevel.T10_REAL_WORLD: ["report", "record"],
        },
        "sibilant_es": {
            TestLevel.T0_RECOGNITION: ["bus", "box"],
            TestLevel.T1_RECALL: ["bus", "box"],
            TestLevel.T3_CONTROLLED_APPLICATION: ["bus", "fox", "wish"],
            TestLevel.T4_INDEPENDENT_APPLICATION: ["glass", "church", "dish"],
            TestLevel.T5_VARIATION: ["class", "brush"],
            TestLevel.T6_ERROR_DETECTION: ["bus", "box"],
            TestLevel.T7_CORRECTION: ["bus", "box"],
            TestLevel.T8_GENERALIZATION: ["watch", "flash"],
            TestLevel.T9_ADVERSARIAL: ["waltz-safe", "match"],
            TestLevel.T10_REAL_WORLD: ["patch", "torch"],
        },
        "y_to_ies": {
            TestLevel.T0_RECOGNITION: ["baby", "city"],
            TestLevel.T1_RECALL: ["baby", "city"],
            TestLevel.T3_CONTROLLED_APPLICATION: ["baby", "city", "party"],
            TestLevel.T4_INDEPENDENT_APPLICATION: ["puppy", "family", "story"],
            TestLevel.T5_VARIATION: ["lady", "berry"],
            TestLevel.T6_ERROR_DETECTION: ["baby", "city"],
            TestLevel.T7_CORRECTION: ["baby", "city"],
            TestLevel.T8_GENERALIZATION: ["company", "county"],
            TestLevel.T9_ADVERSARIAL: ["day-safe", "toy-safe"],
            TestLevel.T10_REAL_WORLD: ["diary", "century"],
        },
    })

    EXPLANATIONS = {
        "add_s": "add s",
        "sibilant_es": "after s, x, ch, sh add es",
        "y_to_ies": HEURISTIC,
    }

    def _lesson_for(self, concept: str) -> Lesson:
        explanation = self.EXPLANATIONS[concept]
        knowledge_class = (KnowledgeClass.HEURISTIC if concept == "y_to_ies"
                           else KnowledgeClass.HARD)
        words = self.WORD_LISTS[concept][TestLevel.T3_CONTROLLED_APPLICATION]
        examples = [f"{w} -> {correct_plural(w)}" for w in words[:2]]
        counterexamples = [f"{w} -> {w}s (wrong: {explanation})" for w in words[:1]]
        return Lesson(
            domain=self.domain, concept=concept,
            objective=f"pluralise words following '{explanation}'",
            explanation=explanation, examples=examples,
            counterexamples=counterexamples,
            knowledge_class=knowledge_class,
            source_knowledge=[explanation])

    def next_lesson(self, profile: AgentProfile,
                    history: Sequence[TestResult]) -> Optional[Lesson]:
        if self.lesson_index >= len(self.concepts):
            return None
        concept = self.concepts[self.lesson_index]
        self.lesson_index += 1
        return self._lesson_for(concept)

    def check_reiteration(self, lesson: Lesson,
                          reiteration: Reiteration) -> ReiterationCheck:
        restate_ok = bool(reiteration.restate) and \
            reiteration.restate == lesson.explanation
        explain_ok = bool(reiteration.explain)
        connect_ok = bool(reiteration.connect)
        example_ok = bool(reiteration.example)
        counterexample_ok = bool(reiteration.counterexample)
        notes = [] if restate_ok else ["restatement did not match the taught rule"]
        return ReiterationCheck(
            restate_ok=restate_ok, explain_ok=explain_ok, connect_ok=connect_ok,
            example_ok=example_ok, counterexample_ok=counterexample_ok,
            notes=notes)

    def _get_or_create_test(self, lesson: Lesson, level: TestLevel, word: str,
                            split: Split) -> TestSpec:
        key = (lesson.concept, level, word)
        cached = self._test_cache.get(key)
        if cached is not None:
            return cached
        test = TestSpec(
            capability=lesson.concept, level=level,
            difficulty=0.3 + 0.07 * int(level), objective=True,
            expected=correct_plural(word), split=split, lesson_id=lesson.id,
            failure_mode_targeted="wrong_suffix",
            payload={"word": word, "question": f"plural of {word}"})
        self._test_cache[key] = test
        self._word_by_test_id[test.id] = word
        return test

    def build_tests(self, lesson: Lesson, profile: AgentProfile,
                    history: Sequence[TestResult]) -> List[TestSpec]:
        """Novelty = 1.0 for a word never given to this agent before, 0.0
        otherwise - tracked through the word each test id was created for,
        so it reflects this agent's actual history, not a guess."""
        seen_words = {self._word_by_test_id.get(r.test_id) for r in history
                     if r.agent_id == profile.id}
        seen_words.discard(None)
        words = self.WORD_LISTS.get(lesson.concept, {})
        out: List[TestSpec] = []
        for level, word_list in words.items():
            for i, word in enumerate(word_list):
                split = Split.HIDDEN if level == TestLevel.T10_REAL_WORLD else (
                    Split.VALIDATION if i == len(word_list) - 1 else Split.TRAINING)
                test = self._get_or_create_test(lesson, level, word, split)
                test.novelty = 0.0 if word in seen_words else 1.0
                out.append(test)
        return out

    def grade(self, test: TestSpec, performance: Performance) -> TestResult:
        word = str(test.payload.get("word", ""))
        expected = correct_plural(word)
        passed = performance.claim.strip().lower() == expected.lower()
        score = 1.0 if passed else 0.0
        failure_mode = "" if passed else test.failure_mode_targeted
        return TestResult(
            test_id=test.id, level=test.level, split=test.split,
            score=score, passed=passed, student_claim=performance.claim,
            student_confidence=performance.confidence,
            trainer_claim=expected, trainer_confidence=0.9,
            failure_mode=failure_mode, evidence=[f"expected: {expected}"])

    def remediate(self, profile: AgentProfile, lesson: Lesson,
                  failures: Sequence[TestResult]) -> Remediation:
        """Alternates between guided instruction and a genuinely different
        word list - never repeats the same test, and mutates the concept's
        word list on the "different_practice" branch so the next
        ``build_tests`` call really does offer new material."""
        self.remediation_calls += 1
        level = failures[0].level if failures else TestLevel.T3_CONTROLLED_APPLICATION
        if self.remediation_calls % 2 == 1:
            return Remediation(
                kind="guided", lesson_id=lesson.id,
                detail=f"walk through the rule with a worked example at "
                       f"{level.label}")
        pool = self.WORD_LISTS.setdefault(lesson.concept, {})
        current = pool.get(level, [])
        pool[level] = [f"{w}2" for w in current] or ["extra"]
        return Remediation(
            kind="different_practice", lesson_id=lesson.id,
            detail=f"practising with a new word list at {level.label}",
            next_level=level)

    def learn_from(self, result: TestResult) -> None:
        self.learned_results.append(result)


@dataclass
class HardRule:
    """The plural-rules hard rule: applies to any dispute whose question is
    "plural of <word>", and decides from the rule table - never the y-heuristic,
    which is defeasible and does not get a vote here."""
    rule_name: str = "plural_hard_rule"

    @property
    def name(self) -> str:
        return self.rule_name

    @property
    def knowledge_class(self) -> KnowledgeClass:
        return KnowledgeClass.HARD

    def applies(self, dispute: Dispute) -> bool:
        return dispute.question.startswith("plural of")

    def decide(self, dispute: Dispute) -> Optional[Ruling]:
        word = dispute.question[len("plural of "):].strip()
        if not word:
            return None
        expected = correct_plural(word)
        suffix = _matched_suffix(word)
        if suffix == "default" and word.endswith("y"):
            # The y -> ies case is heuristic, not hard: a word like "toy"
            # (vowel before y) keeps "s", which the hard table alone cannot
            # tell apart from "city" -> "cities" without the heuristic.
            # The hard rule only speaks when a hard suffix (s/x/ch/sh) or the
            # plain default applies; y-final words are left to the heuristic.
            return None
        accepted = ("student" if dispute.student_claim.lower() == expected.lower()
                   else ("trainer" if dispute.trainer_claim.lower() == expected.lower()
                        else "neither"))
        rejected = "trainer" if accepted == "student" else (
            "student" if accepted == "trainer" else "")
        return Ruling(
            ruling=expected, accepted_claim=accepted, rejected_claim=rejected,
            rationale=f"hard rule '{RULES[suffix]}' gives '{expected}' for '{word}'",
            confidence=1.0,
            correction_student="" if accepted == "student" else
            f"the plural of {word} is {expected} ({RULES[suffix]})",
            correction_trainer="" if accepted == "trainer" else
            f"the plural of {word} is {expected} ({RULES[suffix]})")
