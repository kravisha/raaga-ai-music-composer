"""Who plays a part, decided once.

Two paths used to choose a lead instrument and choose it differently: the
audition read the brief then a configured default then fell back to the
veena; the arrangement read the brief then ranked the catalogue against
the feel of the brief then fell back to the flute.  A creator could hear a
raaga auditioned on one instrument and the arrangement built on another,
from the same brief, with nothing saying why.
"""
from __future__ import annotations

import pytest

from raagacomposer.music import casting
from raagacomposer.music.casting import (FROM_BRIEF, FROM_DEFAULT, FROM_FEEL,
                                         FROM_SETTING)

pytestmark = pytest.mark.unit


def test_what_the_creator_asked_for_wins():
    chosen = casting.cast("lead", preferred=["violin"], feel_words=["sad"])
    assert chosen.instrument.name.lower() == "violin"
    assert chosen.source == FROM_BRIEF
    assert chosen.chosen_by_the_creator
    assert "asked for" in chosen.reason


def test_a_preference_that_cannot_play_the_part_is_not_forced_into_it():
    """"Prefer mridangam" is about the arrangement, not the melody."""
    chosen = casting.cast("lead", preferred=["mridangam"], default="veena")
    assert chosen.instrument.supports("lead")
    assert chosen.source != FROM_BRIEF


def test_a_configured_default_comes_next():
    chosen = casting.cast("lead", preferred=[], configured="flute")
    assert chosen.instrument.name.lower().endswith("flute")
    assert chosen.source == FROM_SETTING


def test_the_feel_of_the_brief_decides_when_nothing_was_named():
    chosen = casting.cast("lead", feel_words=["sad", "longing", "lyrical"])
    assert chosen.instrument.supports("lead")
    assert chosen.source == FROM_FEEL
    assert "suits" in chosen.reason


def test_the_default_is_used_and_says_it_is_the_default():
    chosen = casting.cast("lead", default="veena")
    assert chosen.instrument.name.lower() == "veena"
    assert chosen.source == FROM_DEFAULT
    assert "default" in chosen.reason


def test_an_avoided_instrument_is_never_cast():
    """Avoiding it must outrank preferring it - the creator said both."""
    chosen = casting.cast("lead", preferred=["violin"], avoided=["violin"],
                          default="veena")
    assert chosen.instrument.name.lower() != "violin"


def test_every_choice_can_say_why():
    for kwargs in ({"preferred": ["violin"]},
                   {"configured": "flute"},
                   {"feel_words": ["bright", "festive"]},
                   {}):
        chosen = casting.cast("lead", default="veena", **kwargs)
        assert chosen.reason, f"no reason given for {kwargs}"
        assert chosen.describe().startswith(chosen.instrument.name)


def test_a_bad_ranker_does_not_stop_a_part_being_cast():
    """The controller's ranker asks a language model; it may fail."""
    def explode(*_a, **_k):
        raise RuntimeError("the model is down")

    chosen = casting.cast("lead", feel_words=["sad"], default="veena",
                          rank=explode)
    assert chosen.instrument.name.lower() == "veena"
    assert chosen.source == FROM_DEFAULT
