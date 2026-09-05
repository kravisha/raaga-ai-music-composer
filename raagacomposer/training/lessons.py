"""A studied source becomes lessons the agent can be tested on.

The missing link between the Training tab and the Agent Factory.  Training
already searches, gets the creator's approval, ingests one source at a time
and writes a Learning Report; the Factory already builds a test ladder from a
``Lesson`` and refuses to call an agent proficient without a graded result.
Nothing joined the two: lessons came only from the shipped curriculum, so
everything a source taught sat in the report and was never examined on.

This turns a completed report into framework ``Lesson`` objects.

**Every lesson made here is stated knowledge, and says so.**  A transcript is
a person telling us something; it is not something the agent heard.  The
project's rule (``docs/DECISIONS.md``, "Heard and stated are different kinds
of evidence") is that stated material may be stored and shown but must not
reach the music, and it is kept here in three ways:

* the lessons are written to the factory store, which the composer does not
  read, and never to ``repo.phrases``, which it does;
* they carry ``KnowledgeClass.HEURISTIC`` and an ``origin`` naming the video,
  so a dispute against the library loses to a hard rule;
* they declare ``T0`` to ``T2`` as their test ceiling - recognition, recall
  and explanation - because an application test built from a transcript would
  grade the agent on something nobody verified by ear.

The consequence is deliberate and is the honest answer to "is it proficient
yet": a concept taught only from transcripts can reach **L3, can explain**,
and no further.  ``factory/mastery.py`` already enforces exactly that cap
without being told about any of this.  Passing a quiz about Sindhu Bhairavi
is not being able to play it.
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..factory.models import KnowledgeClass, Lesson, TestLevel
from .models import LearningReport, LearningSource

log = get_logger("training.lessons")

#: What a stated lesson may be tested at.  Recognition, recall, explanation.
STATED_LEVELS = (TestLevel.T0_RECOGNITION, TestLevel.T1_RECALL,
                 TestLevel.T2_EXPLANATION)

#: The mark every lesson from this module carries.
STATED = "stated"

DOMAIN = "carnatic-music"

#: An objective's category maps to the concept a lesson is about, so two
#: videos teaching arohanam produce one concept with two sources rather than
#: two concepts that never meet.
_CONCEPT_WORDS = (
    ("arohanam", "arohanam"), ("avarohanam", "avarohanam"),
    ("aarohanam", "arohanam"), ("scale", "arohanam"),
    ("prayoga", "prayogas"), ("phrase", "prayogas"),
    ("sangathi", "prayogas"),
    ("gamaka", "gamaka"), ("ornament", "gamaka"),
    ("tala", "tala"), ("rhythm", "tala"), ("beat", "tala"),
    ("nyasa", "nyasa"), ("resting", "nyasa"),
    ("jeeva", "jeeva"), ("important", "jeeva"),
    ("alapana", "alapana"), ("improvis", "alapana"),
    ("mistake", "common_mistakes"), ("confus", "common_mistakes"),
    ("varnam", "compositions"), ("kriti", "compositions"),
    ("geetham", "compositions"), ("composition", "compositions"),
)


def concept_for(text: str, raaga: str = "") -> str:
    """The concept a piece of stated material is about.

    Deliberately a small keyword map rather than anything clever: a concept
    name only has to be stable, so that two sources teaching the same thing
    meet on one record and the curriculum can find them.
    """
    low = (text or "").lower()
    topic = "general"
    for needle, name in _CONCEPT_WORDS:
        if needle in low:
            topic = name
            break
    return f"{topic}:{raaga}" if raaga else topic


def _sentences(text: str, limit: int = 6) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text or "")
             if len(p.strip()) > 12]
    return parts[:limit]


def lessons_from_report(report: LearningReport, raaga: str = "",
                        source: Optional[LearningSource] = None
                        ) -> List[Lesson]:
    """Framework lessons for what one studied source said.

    One lesson per concept the source's objectives touched, built from what
    the report already recorded: what was understood, what was learned, and
    the objective that asked for it.  A source that taught nothing produces
    no lessons rather than an empty one - the report already says it taught
    nothing, and a lesson with no content would be something to examine on
    that nobody could answer.
    """
    source = source or report.source
    origin = (source.url or source.title) if source else "a studied source"
    title = (source.title if source else "") or origin

    learned = list(report.learned)
    understood = _sentences(report.understood)
    if not learned and not understood:
        return []

    by_concept: dict = {}
    for objective in report.objectives:
        if not objective.met:
            # Only what the source actually delivered becomes examinable.
            continue
        concept = concept_for(f"{objective.description} {objective.category}",
                              raaga)
        entry = by_concept.setdefault(concept, {"objectives": [],
                                                "evidence": []})
        entry["objectives"].append(objective.description)
        if objective.evidence:
            entry["evidence"].append(objective.evidence)

    if not by_concept:
        # No objective was met, but the source still said things worth
        # examining on; file them under what the material was about.
        by_concept[concept_for(title, raaga)] = {"objectives": [], "evidence": []}

    lessons: List[Lesson] = []
    for concept, entry in by_concept.items():
        objective = "; ".join(entry["objectives"][:3]) or \
            f"what {title} teaches about {concept.split(':')[0]}"
        explanation = " ".join(understood) or " ".join(learned[:3])
        lesson = Lesson(
            domain=DOMAIN,
            concept=concept,
            objective=objective,
            explanation=explanation[:1200],
            examples=[item[:200] for item in learned[:6]],
            source_knowledge=[origin],
            # Stated, defeasible, and beaten by a hard rule in a dispute.
            knowledge_class=KnowledgeClass.HEURISTIC,
            confidence=min(0.7, max(0.25, float(report.confidence or 0.4))),
            origin=f"{STATED}:{origin}",
            expected_behavior=("restate and explain what this source taught; "
                               "playing it is not examined here"),
            common_errors=[c.summary()[:200] for c in report.conflicts[:3]
                           if hasattr(c, "summary")],
        )
        lessons.append(lesson)
    log.info("%d stated lesson(s) from %s", len(lessons), title)
    return lessons


def is_stated(lesson: Lesson) -> bool:
    """Was this lesson built from something a person said rather than heard?"""
    return str(getattr(lesson, "origin", "") or "").startswith(f"{STATED}:")


def levels_for(lesson: Lesson) -> Sequence[TestLevel]:
    """The test levels a lesson may be examined at.

    A stated lesson stops at explanation.  Everything else keeps whatever the
    trainer would have chosen, which is the full ladder.
    """
    return STATED_LEVELS if is_stated(lesson) else ()
