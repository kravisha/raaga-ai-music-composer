"""How a cycle is read off a brief, and how it must not be.

Arya's findings: "chase" matched inside "purchase", and "a lullaby, not a
chase" chose a chase and cited the rejected idea as its reason.
"""
from __future__ import annotations

from raagacomposer.core.models import CreativeBrief
from raagacomposer.music import tala


def suggest(**fields):
    return tala.suggest(CreativeBrief(**fields))


def test_a_word_inside_another_word_is_not_a_match():
    """"chase" is inside "purchase". The library's voice lookup already
    matched on whole words for the same reason; this did not."""
    choice = suggest(situation="a quiet purchase of a gift", mood="")
    assert choice.tala.name != "Khanda Chapu", choice.reason
    assert "chase" not in choice.reason


def test_an_idea_the_brief_rules_out_is_not_a_request():
    choice = suggest(situation="a lullaby, not a chase", mood="")
    assert choice.tala.name == "Tisra Eka", choice.reason
    assert "chase" not in choice.reason, \
        "it cited the very thing the brief rejected"


def test_a_genuine_mention_still_counts_when_another_is_rejected():
    """Rejection is per mention, not per word: "a chase, and later a
    lullaby" does ask for a chase."""
    choice = suggest(situation="a chase, and later a lullaby", mood="")
    assert choice.tala.name == "Khanda Chapu", choice.reason


def test_the_situation_is_read_before_the_mood():
    """Every new brief carries the default mood, which mentions "nervous".
    Merging all the fields let that decide every song in the application."""
    choice = suggest(situation="a lullaby for a child")
    assert choice.tala.name == "Tisra Eka", choice.reason
    assert "situation" in choice.reason


def test_the_reason_names_the_word_and_where_it_was_found():
    choice = suggest(situation="a village folk festival", mood="")
    assert "folk" in choice.reason and "situation" in choice.reason


def test_it_offers_rather_than_rules():
    """No lullaby has to be in three; the wording has to say so."""
    choice = suggest(situation="a lullaby for a child", mood="")
    assert not choice.chosen_by_creator
    text = choice.describe()
    assert text.startswith("I suggest"), text
    assert "Change it" in text
    assert "often" in choice.reason, "stated as a rule rather than a habit"


def test_an_explicit_cycle_is_the_creator_s_and_says_so():
    choice = suggest(situation="a chase through a city", tala="Rupaka")
    assert choice.tala.name == "Rupaka"
    assert choice.chosen_by_creator
    assert choice.describe().startswith("you chose")


def test_nothing_in_the_brief_falls_back_to_adi_and_says_that():
    choice = suggest(situation="", mood="", feel="")
    assert choice.tala.name == "Adi"
    assert "nothing in the brief" in choice.reason
