"""What the application asks a language model for (spec section 11).

There are five such requests, and they are not alike.  Writing a lyric line to
an exact syllable count is hard and its quality is heard by everyone; deciding
that "add veena here" is an ``arrange.add`` is easy, and it has to happen
before the creator finishes speaking.  Sending both to the same model is either
wasteful or too slow.

The router needs to know more about a task than its name, so that knowledge
lives here as data, beside the interface it describes, rather than being
scattered through the adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class Complexity(str, Enum):
    """How much model is actually needed to do the job well."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    complexity: Complexity
    max_tokens: int
    latency_critical: bool = False
    quality_critical: bool = False
    wants_json: bool = True
    description: str = ""


WRITE_LYRICS = "write_lyrics"
CLASSIFY_INTENT = "classify_intent"
SUGGEST_RAAGAS = "suggest_raagas"
SUGGEST_INSTRUMENTS = "suggest_instruments"
EXPLAIN = "explain"


TASKS: Dict[str, TaskSpec] = {
    # Creative, tightly constrained, and the result is sung aloud.  Worth the
    # strongest model available.
    WRITE_LYRICS: TaskSpec(
        WRITE_LYRICS, Complexity.HIGH, 2000, quality_critical=True,
        description="write lines to an exact syllable count and stress pattern"),
    # Musical judgement over a closed list.  A weak model produces plausible
    # nonsense here, which is worse than no answer at all.
    SUGGEST_RAAGAS: TaskSpec(
        SUGGEST_RAAGAS, Complexity.HIGH, 800, quality_critical=True,
        description="rank raagas for a creative brief, with reasons"),
    # A few sentences of practical advice.  Wrong is recoverable.
    EXPLAIN: TaskSpec(
        EXPLAIN, Complexity.MEDIUM, 400, wants_json=False,
        description="answer a musical question in a few sentences"),
    # Pick from a catalog that is supplied in the prompt.
    SUGGEST_INSTRUMENTS: TaskSpec(
        SUGGEST_INSTRUMENTS, Complexity.LOW, 300,
        description="choose instruments from a closed catalog"),
    # Only reached when the rule tables have already failed, and the creator is
    # mid-sentence.  A local model answering in 200ms beats a better one
    # answering in two seconds.
    CLASSIFY_INTENT: TaskSpec(
        CLASSIFY_INTENT, Complexity.LOW, 300, latency_critical=True,
        description="map a spoken instruction onto one of a closed intent set"),
}


def spec(name: str) -> TaskSpec:
    """The spec for ``name``, or a middling default for an unknown task."""
    return TASKS.get(name, TaskSpec(name, Complexity.MEDIUM, 600))
