"""YouTube as a curriculum source - docs/PLAN_youtube_curriculum.md, Y1 and Y2.

The creator's brief: find videos and approve them, extract structured lessons
into the knowledge base, turn those into quizzes, and only call the agent
proficient when it passes.  These cover the first two steps, and the rule
that runs through all four - a transcript is something a person *said*, so it
may be examined on and must never reach the music.
"""
from __future__ import annotations

import pytest

from raagacomposer.factory.models import KnowledgeClass, TestLevel
from raagacomposer.training import lessons as lesson_builder
from raagacomposer.training import youtube
from raagacomposer.training.models import (LearningReport, LearningSource,
                                           Objective, ObjectiveStatus)

pytestmark = pytest.mark.unit

VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# -- Y1: finding, without fetching -----------------------------------------
def test_every_shape_of_youtube_link_is_recognised():
    assert youtube.video_ids(VIDEO) == ["dQw4w9WgXcQ"]
    assert youtube.video_ids("https://youtu.be/dQw4w9WgXcQ") == ["dQw4w9WgXcQ"]
    assert youtube.video_ids(
        "https://www.youtube.com/shorts/dQw4w9WgXcQ") == ["dQw4w9WgXcQ"]
    assert youtube.video_ids(
        "https://www.youtube.com/embed/dQw4w9WgXcQ") == ["dQw4w9WgXcQ"]


def test_ordinary_words_are_not_mistaken_for_videos():
    """An id is eleven characters of a known alphabet; a search phrase is not."""
    assert youtube.video_ids("Sindhu Bhairavi alapana lesson") == []
    assert youtube.video_ids("") == []
    assert youtube.video_ids("youtube.com/watch?v=tooshort") == []


def test_several_links_keep_their_order_and_do_not_repeat():
    text = f"{VIDEO} and https://youtu.be/aBcDeFgHiJk and {VIDEO} again"
    assert youtube.video_ids(text) == ["dQw4w9WgXcQ", "aBcDeFgHiJk"]


def test_a_pasted_link_becomes_a_lead_without_touching_the_network():
    leads = youtube.leads_from_text(VIDEO, allow_network=False)
    assert len(leads) == 1
    assert leads[0]["url"] == VIDEO
    assert leads[0]["video_id"] == "dQw4w9WgXcQ"
    # It says what it is rather than inventing a title it did not look up.
    assert "dQw4w9WgXcQ" in leads[0]["title"]


def test_the_finder_needs_no_key_for_pasted_links(monkeypatch):
    from raagacomposer.core.settings import Settings

    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, n: ""))
    settings = Settings()
    settings.training_allow_web = False
    find = youtube.finder(settings)
    assert len(find(f"study this {VIDEO}", 10)) == 1


def test_a_phrase_finds_nothing_here_without_a_key(monkeypatch):
    """Not a failure - the other providers answer instead, exactly as they
    do today for a creator who never configures anything."""
    from raagacomposer.core.settings import Settings

    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, n: ""))
    settings = Settings()
    settings.training_allow_web = True
    assert youtube.finder(settings)("sindhu bhairavi lesson", 10) == []


def test_a_phrase_is_not_searched_when_the_web_is_switched_off(monkeypatch):
    from raagacomposer.core.settings import Settings

    monkeypatch.setattr(Settings, "secret",
                        classmethod(lambda cls, n: "a-key"))
    called = []
    monkeypatch.setattr(youtube, "search_api",
                        lambda *a, **k: called.append(a) or [])
    settings = Settings()
    settings.training_allow_web = False
    assert youtube.finder(settings)("sindhu bhairavi", 10) == []
    assert not called, "nothing may be searched while the web is off"


# -- Y2: a source becomes lessons ------------------------------------------
def _report(**over) -> LearningReport:
    source = LearningSource(title="Sindhu Bhairavi alapana", url=VIDEO,
                            source_type="lead")
    fields = dict(
        source=source, summary="A lesson on Sindhu Bhairavi.",
        understood=("Sindhu Bhairavi takes both R1 and R2. "
                    "The descent uses D1 where the ascent uses D2."),
        learned=["arohanam S R2 G2 M1 P D2 N2 S",
                 "avarohanam S N2 D1 P M1 G2 R1 S"],
        confidence=0.55,
        objectives=[
            Objective(description="Learn arohanam and avarohanam",
                      category="scale", status=ObjectiveStatus.LEARNED,
                      confidence=0.6, evidence="stated at 2:10"),
            Objective(description="Identify characteristic prayogas",
                      category="phrases", status=ObjectiveStatus.LEARNED,
                      confidence=0.5),
            Objective(description="Identify the tala", category="tala",
                      status=ObjectiveStatus.NOT_PRESENT),
        ])
    fields.update(over)
    return LearningReport(**fields)


def test_a_studied_source_becomes_one_lesson_per_concept_it_taught():
    made = lesson_builder.lessons_from_report(_report(), raaga="Sindhu Bhairavi")
    concepts = {lesson.concept for lesson in made}
    assert concepts == {"arohanam:Sindhu Bhairavi", "prayogas:Sindhu Bhairavi"}


def test_an_objective_the_source_did_not_meet_teaches_nothing():
    """The report already says the tala was not present; a lesson about it
    would be something to examine on that nobody could answer."""
    made = lesson_builder.lessons_from_report(_report(), raaga="Sindhu Bhairavi")
    assert not any("tala" in lesson.concept for lesson in made)


def test_a_source_that_taught_nothing_produces_no_lessons():
    empty = _report(learned=[], understood="", objectives=[])
    assert lesson_builder.lessons_from_report(empty, raaga="X") == []


def test_every_lesson_is_marked_stated_and_traceable_to_its_video():
    for lesson in lesson_builder.lessons_from_report(_report(),
                                                     raaga="Sindhu Bhairavi"):
        assert lesson_builder.is_stated(lesson)
        assert lesson.origin.startswith("stated:")
        assert VIDEO in lesson.origin or VIDEO in lesson.source_knowledge
        # Defeasible: a hard rule from the library beats it in a dispute.
        assert lesson.knowledge_class is KnowledgeClass.HEURISTIC


def test_a_stated_lesson_is_examined_at_most_on_explaining_it():
    """An application test built from a transcript would grade the agent on
    something nobody verified by ear."""
    for lesson in lesson_builder.lessons_from_report(_report(),
                                                     raaga="Sindhu Bhairavi"):
        assert tuple(lesson_builder.levels_for(lesson)) == (
            TestLevel.T0_RECOGNITION, TestLevel.T1_RECALL,
            TestLevel.T2_EXPLANATION)


def test_two_videos_teaching_one_thing_meet_on_one_concept():
    """Otherwise the curriculum could never see that a topic was covered
    twice, and every video would be its own island."""
    first = lesson_builder.lessons_from_report(_report(), raaga="Kalyani")
    second = lesson_builder.lessons_from_report(_report(), raaga="Kalyani")
    assert {lesson.concept for lesson in first} == \
        {lesson.concept for lesson in second}


def test_confidence_is_bounded_by_what_a_transcript_can_be_worth():
    generous = _report(confidence=1.0)
    for lesson in lesson_builder.lessons_from_report(generous, raaga="X"):
        assert lesson.confidence <= 0.7
