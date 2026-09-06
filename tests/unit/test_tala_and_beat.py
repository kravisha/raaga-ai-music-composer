"""Tala as a real cycle, and a beat you can make and vary.

Percussion existed only as a side effect of arranging the whole song: you
could not make a beat, hear one without a full mix, or change one without
rebuilding the arrangement around it.  Rhythm was an integer -
``beats_per_cycle`` - and a tala is not a number of beats.  It is a named
cycle with an internal shape, and that shape is where the accents fall.
"""
from __future__ import annotations

import pytest

from raagacomposer.core.models import BeatVersion
from raagacomposer.music import beat as beat_engine
from raagacomposer.music import tala as tala_module

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# tala
# --------------------------------------------------------------------------
def test_every_tala_adds_up():
    """A cycle whose angas do not sum to its aksharas is not a cycle."""
    for tala in tala_module.TALAS:
        assert sum(tala.angas) == tala.aksharas, tala.name


def test_a_broken_tala_refuses_to_exist():
    with pytest.raises(ValueError):
        tala_module.Tala("Nonsense", 8, (3, 3))


def test_accents_come_from_the_shape_not_a_second_list():
    """Adi is 4+2+2, so the weight falls on beats 0, 4 and 6."""
    adi = tala_module.require("Adi")
    assert adi.accents == (0, 4, 6)
    assert adi.is_accent(0) and adi.is_accent(4) and not adi.is_accent(1)

    misra = tala_module.require("Misra Chapu")
    assert misra.aksharas == 7 and misra.accents == (0, 3, 5)


def test_a_tala_is_found_by_the_names_people_use():
    for spelling in ("adi", "Adi", "adi tala", "chaturasra triputa"):
        assert tala_module.find(spelling).name == "Adi"
    assert tala_module.find("not a tala") is None


def test_an_existing_tune_says_its_tala_by_its_beat_count():
    assert tala_module.for_beats(8).name == "Adi"
    assert tala_module.for_beats(7).name == "Misra Chapu"
    assert tala_module.for_beats(99).name == tala_module.DEFAULT_TALA


def test_a_pattern_accents_the_start_of_every_anga():
    misra = tala_module.require("Misra Chapu")
    beats = {offset for offset, _s, _v in tala_module.pattern(misra, "steady")}
    for accent in misra.accents:
        assert float(accent) in beats, f"beat {accent} is not played"
    loudest = max(tala_module.pattern(misra, "steady"), key=lambda s: s[2])
    assert loudest[0] == 0.0, "the cycle should start on its heaviest stroke"


def test_density_changes_how_much_is_played_not_where_the_weight_is():
    adi = tala_module.require("Adi")
    counts = {d: len(tala_module.pattern(adi, d)) for d in tala_module.DENSITIES}
    assert counts["sparse"] < counts["steady"] < counts["busy"]
    for density in tala_module.DENSITIES:
        played = {o for o, _s, _v in tala_module.pattern(adi, density)}
        for accent in adi.accents:
            assert float(accent) in played, f"{density} dropped an accent"


# --------------------------------------------------------------------------
# the beat
# --------------------------------------------------------------------------
def test_a_beat_fills_the_time_it_was_given():
    adi = tala_module.require("Adi")
    notes = beat_engine.generate(adi, 120, 16.0, seed=3)
    assert notes, "no strokes"
    assert max(n.start for n in notes) < 16.0
    assert max(n.start for n in notes) > 12.0, "the last cycles are missing"
    assert all(n.start >= 0 for n in notes)


def test_the_beat_lines_up_with_the_cycle():
    """Strokes land on the tala's grid, so melody and beat agree by
    construction rather than by being nudged into alignment."""
    from raagacomposer.music.theory import beat_seconds

    adi = tala_module.require("Adi")
    beat = beat_seconds(96)
    notes = beat_engine.generate(adi, 96, 12.0, seed=1, density="steady")
    for note in notes:
        beats_in = note.start / beat
        assert abs(beats_in - round(beats_in * 4) / 4) < 1e-6, \
            f"stroke at {note.start}s is off the grid"


def test_a_variation_is_the_same_beat(monkeypatch):
    """Its tala never moves - a beat whose accents have moved is a
    different beat, not a variation of this one."""
    first = beat_engine.realise(BeatVersion(
        tala="Misra Chapu", tempo_bpm=100, duration=10.0, seed=4))
    for strength in beat_engine.STRENGTHS:
        varied = beat_engine.realise(
            beat_engine.vary(first, strength=strength, seed=9))
        assert varied.tala == first.tala, f"{strength} changed the tala"
        assert varied.tempo_bpm == first.tempo_bpm
        assert varied.parent_version == first.version
        assert varied.version == first.version + 1
        assert varied.notes, f"{strength} produced silence"


def test_only_a_twist_may_change_the_density():
    first = beat_engine.realise(BeatVersion(tala="Adi", tempo_bpm=90,
                                            duration=8.0, density="steady",
                                            seed=2))
    for strength in ("subtle", "moderate"):
        assert beat_engine.vary(first, strength=strength).density == "steady"
    assert beat_engine.vary(first, strength="twist", seed=5).density != "steady"


def test_a_variation_actually_differs():
    first = beat_engine.realise(BeatVersion(tala="Adi", tempo_bpm=90,
                                            duration=8.0, seed=2))
    second = beat_engine.realise(beat_engine.vary(first, strength="moderate",
                                                  seed=77))
    a = [(n.start, n.velocity, n.midi) for n in first.notes]
    b = [(n.start, n.velocity, n.midi) for n in second.notes]
    assert a != b, "the variation is identical to what it varied"
