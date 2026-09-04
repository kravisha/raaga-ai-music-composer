"""The creator's words in the evaluator's vocabulary (item 4 of
docs/PLAN_learning_loop.md): a complaint about a tune the critic passed
still becomes guidance the next tune obeys."""
from __future__ import annotations

import pytest

from raagacomposer.agent.music_agent import MusicAgent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("text, expected", [
    ("This does not sound like Keeravani, too mechanical.",
     {"neighbour_drift", "no_idiom", "repetitive", "no_gamaka"}),
    ("Too flat, no ornament at all.", {"no_gamaka"}),
    ("It jumps all over the place.", {"too_many_leaps"}),
    ("It ends abruptly and never resolves.", {"no_cadence"}),
    ("The timing is off the beat.", {"off_beat"}),
    ("Lovely, keep it.", set()),
])
def test_feedback_maps_to_finding_kinds(text, expected):
    assert set(MusicAgent.feedback_kinds(text)) == expected


def test_feedback_kinds_are_deduplicated_and_ordered():
    kinds = MusicAgent.feedback_kinds("boring and repetitive and monotonous")
    assert kinds == ["repetitive", "no_gamaka"]
