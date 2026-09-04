"""Unit tests: guidance, the context builder that turns lessons into
constraints for the next attempt (learning specification section 15 #9,
docs/PLAN_learning_loop.md "A context builder turns lessons into guidance").

``guidance_from_lessons`` is a pure function of a list of ``Lesson`` rows, so
most of this file constructs lessons directly rather than going through a
repository; ``build_guidance`` is the one function that reads a repository,
so it gets its own small fixture.
"""
from __future__ import annotations

import random

import pytest

from raagacomposer.agent.guidance import (Guidance, build_guidance,
                                          guidance_from_lessons)
from raagacomposer.agent.knowledge import KnowledgeRepository, Lesson

pytestmark = pytest.mark.unit


@pytest.fixture
def repo(tmp_path) -> KnowledgeRepository:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


def _lesson(kind: str, evidence: str = "", related=None, recurrences: int = 1,
           unit_id: str = "b13.short_phrase:Keeravani",
           raaga: str = "Keeravani", applied: bool = False) -> Lesson:
    return Lesson(raaga=raaga, unit_id=unit_id, kind=kind, evidence=evidence,
                 related=list(related or []), recurrences=recurrences,
                 applied=applied)


# --------------------------------------------------------------------------
# kind -> constraint mapping
# --------------------------------------------------------------------------
def test_outside_and_forbidden_swara_become_avoid_swaras():
    lessons = [_lesson("outside_swara", evidence="G3 M2"),
               _lesson("forbidden_swara", evidence="N1")]
    guidance = guidance_from_lessons(lessons)
    assert guidance.avoid_swaras == {"G3", "M2", "N1"}
    assert "outside_swara" in guidance.kinds
    assert "forbidden_swara" in guidance.kinds


def test_wrong_direction_becomes_two_avoid_transitions():
    guidance = guidance_from_lessons(
        [_lesson("wrong_direction", evidence="S>G3 N2>D1")])
    assert guidance.avoid_transitions == {("S", "G3"), ("N2", "D1")}
    assert guidance.kinds == ["wrong_direction"]


def test_no_cadence_forces_a_nyasa_ending_and_avoids_the_bad_one():
    guidance = guidance_from_lessons([_lesson("no_cadence", evidence="D1")])
    assert guidance.must_end_on_nyasa is True
    assert guidance.avoid_endings == {"D1"}


def test_too_many_leaps_raises_prefer_step_with_recurrences():
    guidance = guidance_from_lessons(
        [_lesson("too_many_leaps", recurrences=3)])
    assert guidance.prefer_step == pytest.approx(0.45)


def test_too_many_leaps_prefer_step_is_capped():
    guidance = guidance_from_lessons(
        [_lesson("too_many_leaps", recurrences=4)])
    # 0.15 * 4 = 0.6, capped at MAX_STEP_BIAS (0.5).
    assert guidance.prefer_step == pytest.approx(0.5)


def test_not_original_avoids_quoting_the_related_phrase():
    guidance = guidance_from_lessons(
        [_lesson("not_original", related=["phr-abc123"])])
    assert guidance.avoid_quoting == {"phr-abc123"}


def test_not_original_with_phrases_fills_avoid_runs():
    guidance = guidance_from_lessons(
        [_lesson("not_original", related=["phr-abc123"])],
        phrases={"phr-abc123": ["S", "G2", "M1", "P", "D1"]})
    assert guidance.avoid_runs == {("S", "G2", "M1", "P", "D1")}
    # avoid_quoting is filled either way; avoid_runs only when the phrase's
    # own swaras were supplied.
    assert guidance.avoid_quoting == {"phr-abc123"}


def test_not_original_without_phrases_leaves_avoid_runs_empty():
    guidance = guidance_from_lessons(
        [_lesson("not_original", related=["phr-abc123"])])
    assert guidance.avoid_runs == set()


def test_not_original_ignores_octave_marks_when_filling_avoid_runs():
    guidance = guidance_from_lessons(
        [_lesson("not_original", related=["phr-abc123"])],
        phrases={"phr-abc123": ["S+", "N3", "D1-"]})
    assert guidance.avoid_runs == {("S", "N3", "D1")}


def test_no_idiom_raises_quote_more_and_is_capped():
    guidance = guidance_from_lessons([_lesson("no_idiom", recurrences=1)])
    assert guidance.quote_more == pytest.approx(0.1)
    capped = guidance_from_lessons([_lesson("no_idiom", recurrences=5)])
    # 0.1 * 5 = 0.5, capped at MAX_QUOTE_BIAS (0.4).
    assert capped.quote_more == pytest.approx(0.4)


def test_repetitive_no_gamaka_neighbour_drift_are_flags():
    guidance = guidance_from_lessons([
        _lesson("repetitive"), _lesson("no_gamaka"), _lesson("neighbour_drift"),
    ])
    assert guidance.vary_more is True
    assert guidance.add_gamaka is True
    assert guidance.prefer_jeeva is True


def test_exercise_and_creator_feedback_kinds_contribute_nothing():
    guidance = guidance_from_lessons([
        _lesson("exercise:name the swara"), _lesson("creator_feedback"),
    ])
    assert guidance.is_empty()
    assert guidance.kinds == []


# --------------------------------------------------------------------------
# weighting
# --------------------------------------------------------------------------
def test_a_lesson_of_the_unit_itself_weighs_double():
    same_unit = guidance_from_lessons(
        [_lesson("too_many_leaps", recurrences=1, unit_id="b13.short_phrase:Keeravani")],
        unit_id="b13.short_phrase:Keeravani")
    other_unit = guidance_from_lessons(
        [_lesson("too_many_leaps", recurrences=1, unit_id="b14.chains:Keeravani")],
        unit_id="b13.short_phrase:Keeravani")
    assert same_unit.prefer_step == pytest.approx(2 * other_unit.prefer_step)


def test_applied_lessons_are_skipped():
    guidance = guidance_from_lessons(
        [_lesson("outside_swara", evidence="G3", applied=True)])
    assert guidance.is_empty()
    assert guidance.lesson_ids == []


def test_order_of_input_lessons_does_not_change_the_result():
    lessons = [
        _lesson("outside_swara", evidence="G3"),
        _lesson("wrong_direction", evidence="S>G3"),
        _lesson("no_cadence", evidence="D1"),
        _lesson("too_many_leaps", recurrences=2),
        _lesson("no_idiom"),
        _lesson("repetitive"),
    ]
    forward = guidance_from_lessons(lessons)
    shuffled = list(lessons)
    random.Random(1).shuffle(shuffled)
    backward = guidance_from_lessons(shuffled)
    assert forward.avoid_swaras == backward.avoid_swaras
    assert forward.avoid_transitions == backward.avoid_transitions
    assert forward.avoid_endings == backward.avoid_endings
    assert forward.must_end_on_nyasa == backward.must_end_on_nyasa
    assert forward.prefer_step == backward.prefer_step
    assert forward.quote_more == backward.quote_more
    assert forward.vary_more == backward.vary_more
    assert forward.kinds == backward.kinds
    assert forward.lesson_ids == backward.lesson_ids


# --------------------------------------------------------------------------
# build_guidance
# --------------------------------------------------------------------------
def test_build_guidance_with_no_lessons_is_empty(repo):
    guidance = build_guidance(repo, "Keeravani", "b13.short_phrase:Keeravani")
    assert guidance.is_empty()


def test_build_guidance_with_no_raaga_is_empty(repo):
    assert build_guidance(repo, "", "b13.short_phrase:Keeravani").is_empty()


def test_build_guidance_reads_stored_lessons(repo):
    repo.add_lesson(_lesson("outside_swara", evidence="G3"))
    guidance = build_guidance(repo, "Keeravani", "b13.short_phrase:Keeravani")
    assert not guidance.is_empty()
    assert guidance.avoid_swaras == {"G3"}


# --------------------------------------------------------------------------
# Guidance itself
# --------------------------------------------------------------------------
def test_describe_is_non_empty_when_not_empty():
    guidance = guidance_from_lessons([_lesson("outside_swara", evidence="G3")])
    assert guidance.describe().strip()


def test_describe_is_empty_string_when_empty():
    assert Guidance().describe() == ""


def test_allows_transition_ignores_octave_marks():
    guidance = Guidance(avoid_transitions={("S", "G3")})
    assert guidance.allows_transition("S+", "G3+") is False
    assert guidance.allows_transition("S-", "G3") is False
    assert guidance.allows_transition("S", "G2") is True


def test_allows_ending_ignores_octave_marks():
    guidance = Guidance(avoid_endings={"D1"})
    assert guidance.allows_ending("D1+") is False
    assert guidance.allows_ending("D1-") is False
    assert guidance.allows_ending("N2") is True


def test_replays_is_false_with_no_avoid_runs():
    assert Guidance().replays(["S", "G2", "M1"]) is False


def test_replays_finds_a_matching_run_ignoring_octave_marks():
    guidance = Guidance(avoid_runs={("S", "G2", "M1", "P")})
    assert guidance.replays(["S+", "G2-", "M1"]) is True
    assert guidance.replays(["N3", "G2", "M1", "P+"]) is True    # mid-run slice


def test_replays_needs_three_matching_notes_not_two():
    guidance = Guidance(avoid_runs={("S", "G2", "M1", "P")})
    # Only two tokens: too short to ever match the three-note window.
    assert guidance.replays(["S", "G2"]) is False
    # Three tokens, but the last one breaks the run: not a replay.
    assert guidance.replays(["S", "G2", "D1"]) is False


def test_replays_a_run_shorter_than_the_window_never_matches():
    guidance = Guidance(avoid_runs={("S", "G2")})
    assert guidance.replays(["S", "G2", "M1"]) is False
