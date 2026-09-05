"""Quizzes from a studied source - docs/PLAN_youtube_curriculum.md Y3 and Y4.

The creator's step three and four: turn what a video taught into quizzes, and
call the agent proficient only when it passes.  The rule underneath both is
that a transcript is what somebody *said*: it can be examined on to the point
of explaining it, and no further.
"""
from __future__ import annotations

import pytest

from raagacomposer.factory.models import MasteryLevel, TestLevel
from raagacomposer.training.models import (LearningReport, LearningSource,
                                           Objective, ObjectiveStatus)

pytestmark = pytest.mark.integration

VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _report(raaga: str = "Keeravani") -> LearningReport:
    return LearningReport(
        source=LearningSource(title=f"{raaga} alapana", url=VIDEO,
                              source_type="lead"),
        summary=f"A lesson on {raaga}.",
        understood=f"{raaga} rests on S, P and G2.",
        learned=["nyasa S P G2"],
        confidence=0.5,
        objectives=[Objective(description="Learn the resting notes",
                              category="nyasa",
                              status=ObjectiveStatus.LEARNED, confidence=0.6)])


def _taught(app):
    app.agent.file_stated_lessons(_report(), raaga="Keeravani")
    return app.agent.trainer(), app.agent.profile()


def test_a_video_lesson_becomes_a_quiz(app):
    """Step three: what the source taught is turned into questions."""
    trainer, profile = _taught(app)
    lesson = trainer.next_lesson(profile, [])
    assert lesson is not None and lesson.origin.startswith("stated:")

    tests = trainer.build_tests(lesson, profile, [])
    assert tests
    for spec in tests:
        assert spec.payload["skill_type"].endswith(".stated")
        assert spec.level <= TestLevel.T2_EXPLANATION


def test_a_quiz_from_a_transcript_never_asks_the_agent_to_play(app):
    """An application test built from a transcript would grade the agent on
    something nobody verified by ear."""
    trainer, profile = _taught(app)
    lesson = trainer.next_lesson(profile, [])
    for _ in range(6):
        for spec in trainer.build_tests(lesson, profile, []):
            assert spec.level <= TestLevel.T2_EXPLANATION, spec.payload


def test_the_agent_answers_from_what_it_kept_not_from_the_lesson(app):
    """Answering out of the lesson would be reading the answer back."""
    trainer, profile = _taught(app)
    lesson = trainer.next_lesson(profile, [])
    spec = trainer.build_tests(lesson, profile, [])[0]

    student = app.agent.student()
    performance = student.perform(spec)
    assert performance.output
    # It says plainly when it kept nothing, rather than echoing the source.
    if "did not keep" not in performance.output:
        assert "nyasa" in performance.output.lower() or performance.evidence


def test_a_quiz_is_graded_and_recorded(app):
    trainer, profile = _taught(app)
    lesson = trainer.next_lesson(profile, [])
    spec = trainer.build_tests(lesson, profile, [])[0]
    performance = app.agent.student().perform(spec)

    result = trainer.grade(spec, performance)
    assert result.lesson_id == spec.lesson_id
    assert 0.0 <= result.score <= 1.0
    assert result.trainer_claim in ("valid", "invalid")
    if not result.passed:
        assert result.failure_mode == "stated_not_retained"


def test_a_concept_taught_only_from_a_transcript_stops_at_can_explain(app):
    """Step four, and the honest ceiling: passing a quiz about a raaga is
    not being able to play it."""
    from raagacomposer.factory.mastery import apply_evidence
    from raagacomposer.factory.models import MasteryRecord

    trainer, profile = _taught(app)
    lesson = trainer.next_lesson(profile, [])
    record = MasteryRecord(agent_id=profile.id, concept=lesson.concept)

    # Everything a transcript can ever give: restated, explained, and a
    # graded pass at the highest level it may be examined at.
    for kind in ("exposure", "restate", "explain"):
        apply_evidence(record, kind, f"{kind}_1", passed=True)
    apply_evidence(record, "explain", "result_stated_1", passed=True)

    assert record.level <= MasteryLevel.L3_CAN_EXPLAIN, (
        f"a transcript took a concept to {record.level.name}")


def test_the_curriculum_still_advances_once_the_video_is_examined(app):
    """A stated lesson jumps the queue exactly once, when it is new."""
    trainer, profile = _taught(app)
    first = trainer.next_lesson(profile, [])
    assert first.origin.startswith("stated:")

    record = app.agent.factory_store().mastery(profile.id, first.concept)
    record.evidence = ["result_seen_1"]
    app.agent.factory_store().save_mastery(record)

    second = trainer.next_lesson(profile, [])
    assert second is None or not second.origin.startswith("stated:"), \
        "the curriculum must carry on once a video has been examined"
