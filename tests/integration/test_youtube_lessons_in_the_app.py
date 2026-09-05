"""A studied source reaches the agent as lessons - and not as music.

docs/PLAN_youtube_curriculum.md, the join between Y2 and step four of the
creator's brief.  The unit tests prove the lessons are built correctly; these
prove they arrive where they should and nowhere else.
"""
from __future__ import annotations

import pytest

from raagacomposer.training.models import (LearningReport, LearningSource,
                                           Objective, ObjectiveStatus)

pytestmark = pytest.mark.integration

VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def _report() -> LearningReport:
    return LearningReport(
        source=LearningSource(title="Keeravani alapana", url=VIDEO,
                              source_type="lead"),
        summary="A lesson on Keeravani.",
        understood="Keeravani rests on S, P and G2. The N3 pulls up to S.",
        learned=["nyasa S P G2", "the D1 N3 leap is its signature"],
        confidence=0.5,
        objectives=[Objective(description="Learn the resting notes",
                              category="nyasa",
                              status=ObjectiveStatus.LEARNED, confidence=0.6)])


def test_a_studied_source_becomes_lessons_the_agent_holds(app):
    made = app.agent.file_stated_lessons(_report(), raaga="Keeravani")
    assert made

    stored = app.agent.factory_store().lessons(domain="carnatic-music")
    concepts = {lesson.concept for lesson in stored}
    assert any(c.startswith("nyasa:") for c in concepts), concepts


def test_what_a_transcript_said_never_reaches_the_music(app):
    """The project's rule, at the level a listener would notice it: a phrase
    a teacher stated is not a phrase the agent heard, and only heard phrases
    reach the composer."""
    before = app.agent.repo.count_phrases("Keeravani")
    app.agent.file_stated_lessons(_report(), raaga="Keeravani")
    assert app.agent.repo.count_phrases("Keeravani") == before
    assert app.agent.phrase_bank("Keeravani") == []


def test_filing_lessons_never_breaks_the_run(app):
    """A source was still studied even if nothing downstream could use it."""
    broken = LearningReport(source=None, learned=[], understood="")
    assert app.agent.file_stated_lessons(broken, raaga="Keeravani") == []
    app._file_stated_lessons(broken)          # the app-level hook, no raise


def test_lessons_survive_a_restart(app, settings):
    from raagacomposer.app import AppController

    app.agent.file_stated_lessons(_report(), raaga="Keeravani")
    before = len(app.agent.factory_store().lessons(domain="carnatic-music"))
    assert before
    app.close()

    restarted = AppController(settings)
    try:
        after = restarted.agent.factory_store().lessons(domain="carnatic-music")
        assert len(after) == before
        assert all(lesson.origin.startswith("stated:") for lesson in after
                   if "nyasa" in lesson.concept)
    finally:
        restarted.close()
