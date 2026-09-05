"""Hearing a raaga's scale - Stage 1 pack document 05 section 7, plan item S4.

Document 06's mandatory test E is here by name.  The rest hold the audition
to what the pack asks of playback in document 01 section H: functional swara
labels kept, pitches from the stored pitch classes, and never a scale that
collapses into one repeated note.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.raaga import audition
from raagacomposer.raaga.audition import AuditionPlan, is_playable, plan
from raagacomposer.raaga.library import RaagaLibrary, parse_swara

pytestmark = pytest.mark.unit

NO_USER_FILE = Path("no-such-user-raagas.json")

#: Pack document 01 section A: the pitch class of every swara, relative to Sa.
PITCH_CLASS = {"S": 0, "R1": 1, "R2": 2, "R3": 3, "G1": 2, "G2": 3, "G3": 4,
               "M1": 5, "M2": 6, "P": 7, "D1": 8, "D2": 9, "D3": 10,
               "N1": 9, "N2": 10, "N3": 11}


@pytest.fixture(scope="module")
def lib() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE)


# -- pack document 06, mandatory test E ------------------------------------
def test_e_playback_smoke_test(lib):
    """Selecting Keeravani must play 8 ascending swara events and 8
    descending swara events.  There must be changing pitches according to
    the stored pitch classes."""
    heard = plan(lib.require("Keeravani"))

    assert len(heard.ascending) == 8
    assert len(heard.descending) == 8

    assert [n.swara for n in heard.ascending] == "S R2 G2 M1 P D1 N3 S+".split()
    assert [n.swara for n in heard.descending] == "S+ N3 D1 P M1 G2 R2 S".split()

    # Changing pitches, and changing in the direction each half claims.
    up = [n.midi for n in heard.ascending]
    down = [n.midi for n in heard.descending]
    assert up == sorted(up) and len(set(up)) == 8
    assert down == sorted(down, reverse=True) and len(set(down)) == 8

    # ... according to the stored pitch classes.
    for note in heard.notes:
        base, octave = parse_swara(note.swara)
        assert (note.midi - audition.TONIC) == PITCH_CLASS[base] + 12 * octave

    assert is_playable(heard)


# -- what the pack asks of playback ----------------------------------------
def test_every_note_keeps_its_functional_swara(lib):
    """Document 01 section H rule 2: never store only the MIDI note number.
    The label is what makes an audition a swara test rather than a tune."""
    for name in ("Keeravani", "Kalyani", "Shubhapantuvarali", "Mohanam"):
        for note in plan(lib.require(name)).notes:
            assert note.swara
            assert parse_swara(note.swara)[0] in PITCH_CLASS


def test_the_scale_never_collapses_into_one_repeated_note(lib):
    """Document 01 section H rule 7, and the failure an untested
    implementation actually produces."""
    for raaga in lib.all():
        heard = plan(raaga)
        assert is_playable(heard), raaga.name
        assert len(set(heard.pitches)) > 2, raaga.name


def test_it_plays_what_the_library_stores_and_not_a_phrase(lib):
    """Exact is the point: a disagreement should be about the raaga, not
    about the performance."""
    for name in ("Keeravani", "Mohanam", "Hamsadhwani"):
        raaga = lib.require(name)
        heard = plan(raaga)
        assert [n.swara for n in heard.ascending] == list(raaga.arohanam)
        assert [n.swara for n in heard.descending] == list(raaga.avarohanam)


def test_a_janya_is_auditioned_as_itself(lib):
    """Mohanam has five swaras, not seven, and the audition says so rather
    than padding it out to the eight events test E happens to expect."""
    heard = plan(lib.require("Mohanam"))
    assert len(heard.ascending) == len(lib.require("Mohanam").arohanam) == 6
    assert is_playable(heard)


def test_the_two_directions_do_not_run_into_each_other(lib):
    """A breath between them, so they are two phrases rather than one
    sixteen-note scale."""
    heard = plan(lib.require("Keeravani"))
    last_up = heard.ascending[-1]
    first_down = heard.descending[0]
    gap = first_down.start - (last_up.start + last_up.duration)
    assert gap >= audition.TURN_SECONDS - 1e-9


def test_the_tonic_can_be_moved_without_changing_the_swaras(lib):
    """The creator's tonic is a pitch, not a different raaga."""
    raaga = lib.require("Keeravani")
    middle, higher = plan(raaga), plan(raaga, tonic=67)
    assert middle.swaras == higher.swaras
    assert [p + 7 for p in middle.pitches] == higher.pitches


def test_an_empty_plan_is_not_playable():
    assert not is_playable(AuditionPlan(raaga="nothing"))
