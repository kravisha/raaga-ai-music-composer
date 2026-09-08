"""Where a Critic's revision points: sections by name or by time."""
from types import SimpleNamespace

import pytest

from raagacomposer.core.models import SectionKind
from raagacomposer.production.targets import place_revisions

pytestmark = pytest.mark.unit


def _tune():
    mk = lambda name, kind, start, end, locked=False: SimpleNamespace(  # noqa: E731
        id=name.lower().replace(" ", "-"), name=name, kind=kind, start=start,
        end=end, locked=locked)
    sections = [mk("Prelude", SectionKind.PRELUDE, 0.0, 5.5),
                mk("Pallavi", SectionKind.PALLAVI, 5.5, 16.6),
                mk("Anupallavi", SectionKind.ANUPALLAVI, 16.6, 25.0),
                mk("Interlude 1", SectionKind.INTERLUDE, 25.0, 30.5),
                mk("Charanam 1", SectionKind.CHARANAM, 30.5, 41.6),
                mk("Pallavi 2", SectionKind.PALLAVI, 41.6, 50.0),
                mk("Charanam 2", SectionKind.CHARANAM, 50.0, 58.0),
                mk("Outro", SectionKind.OUTRO, 58.0, 62.0)]
    return SimpleNamespace(sections=sections)


def test_a_bare_section_name_is_the_first_section_of_that_kind():
    p = place_revisions(["Melody specialist: reshape the Charanam's opening phrase"], _tune())
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert not p.unplaced and not p.out_of_range


def test_a_numbered_label_is_exact_and_the_spoken_spelling_counts():
    p = place_revisions(["Vary Charanam 2, and differentiate the Anu Pallavi opening"], _tune())
    assert [s.name for s in p.targets] == ["Anupallavi", "Charanam 2"]


def test_a_time_range_lands_on_the_sections_it_covers():
    p = place_revisions(["reshape the S+–G3–R2+ passage at 9.93–11.49 s"], _tune())
    assert [s.name for s in p.targets] == ["Pallavi"]
    p = place_revisions(["from 24.5 to 26.0 s the join is abrupt"], _tune())
    assert [s.name for s in p.targets] == ["Anupallavi", "Interlude 1"]
    p = place_revisions(["the leap around 17.8 seconds"], _tune())
    assert [s.name for s in p.targets] == ["Anupallavi"]


def test_a_time_outside_the_song_is_reported_not_placed():
    p = place_revisions(["the passage at 500-510 s"], _tune())
    assert not p.targets and p.out_of_range == ["500-510 s"]
    assert "outside the song" in " ".join(p.notes)


def test_a_locked_section_is_named_and_left_alone():  # a change is asked of it
    tune = _tune()
    tune.sections[1].locked = True
    p = place_revisions(["give the Pallavi a new motif; develop the Charanam"], tune)
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert p.skipped_locked == ["Pallavi"]
    assert "Pallavi is locked" in " ".join(p.notes)


def test_a_revision_that_names_nothing_is_about_the_whole_tune():
    p = place_revisions(["make it warmer", "more variation"], _tune())
    assert not p.targets and not p.unplaced and p.whole
    assert "whole tune" in " ".join(p.notes)


def test_targets_come_back_in_time_order_and_once():
    p = place_revisions(["fix the Outro", "the Prelude drags", "again the outro"], _tune())
    assert [s.name for s in p.targets] == ["Prelude", "Outro"]


# ----------------------------------------------------------------------
# Arya's review of de0141e: preserve-only and negated references, the
# whole-song cue, ambiguous places, and section boundaries
# ----------------------------------------------------------------------
def test_a_section_asked_to_be_kept_is_not_a_rewrite_target():
    p = place_revisions(["keep the Pallavi unchanged; rewrite the Charanam"], _tune())
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert p.preserved == ["Pallavi"] and not p.unplaced
    assert "Pallavi" in " ".join(p.notes) and "kept" in " ".join(p.notes)


def test_every_way_of_saying_leave_it_alone_is_read_as_such():
    for text in ["retain the Pallavi's motif; develop the Charanam",
                 "leave the Pallavi alone and vary the Charanam",
                 "don't touch the Pallavi, but the Charanam needs a new answer",
                 "the Pallavi works; the Charanam drags",
                 "the Pallavi is fine. Reshape the Charanam",
                 "keep the Pallavi as it is and give the Charanam a lift"]:
        p = place_revisions([text], _tune())
        assert [s.name for s in p.targets] == ["Charanam 1"], text
        assert p.preserved == ["Pallavi"], text


def test_keep_reaches_every_section_it_lists():
    p = place_revisions(["keep the Pallavi and the Anupallavi; rework the Outro"], _tune())
    assert [s.name for s in p.targets] == ["Outro"]
    assert p.preserved == ["Pallavi", "Anupallavi"]


def test_a_preserved_section_that_is_also_locked_is_simply_preserved():
    tune = _tune()
    tune.sections[1].locked = True
    p = place_revisions(["retain the Pallavi's motif; develop the Charanam"], tune)
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert p.preserved == ["Pallavi"] and p.skipped_locked == []


def test_a_locked_section_asked_to_change_is_reported_as_locked():
    tune = _tune()
    tune.sections[1].locked = True
    p = place_revisions(["vary the Pallavi's motif; develop the Charanam"], tune)
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert p.skipped_locked == ["Pallavi"] and p.preserved == []


def test_the_whole_tune_named_is_a_whole_rewrite_not_an_unplaced_one():
    for text in ["rewrite the whole tune around a clearer motif",
                 "the entire melody sits too low; recompose it",
                 "overall the contour is static",
                 "every section needs more movement"]:
        p = place_revisions([text], _tune())
        assert p.whole and not p.targets and not p.unplaced, text


def test_a_remark_that_names_no_place_at_all_is_a_whole_rewrite():
    p = place_revisions(["make it warmer and more inward"], _tune())
    assert p.whole and not p.unplaced and not p.targets


def test_a_place_the_reader_cannot_resolve_is_unresolved_not_a_whole_rewrite():
    for text in ["reshape the phrase after the big leap",
                 "the cadence in bar 3 is weak",
                 "the second passage should answer the first",
                 "the S+-G3-R2+ figure repeats too often"]:
        p = place_revisions([text], _tune())
        assert not p.whole and not p.targets, text
        assert p.unplaced == [text], text
    assert "could not be placed" in " ".join(p.notes)


def test_a_whole_rewrite_spares_the_sections_asked_to_be_kept():
    tune = _tune()
    tune.sections[4].locked = True    # Charanam 1
    p = place_revisions(["keep the Pallavi unchanged, but overall the tune is static"], tune)
    assert not p.whole
    assert [s.name for s in p.targets] == ["Prelude", "Anupallavi", "Interlude 1",
                                           "Pallavi 2", "Charanam 2", "Outro"]
    assert p.preserved == ["Pallavi"]


def test_keeping_everything_rewrites_nothing():
    p = place_revisions(["keep the whole tune as it is"], _tune())
    assert not p.whole and not p.targets and not p.unplaced
    assert p.preserved == ["the whole tune"]


def test_a_sections_own_ending_or_opening_is_that_section_not_the_outro():
    p = place_revisions(["the Pallavi's ending drags; reshape the opening of the Charanam"], _tune())
    assert [s.name for s in p.targets] == ["Pallavi", "Charanam 1"]
    p = place_revisions(["the ending of the Pallavi drags"], _tune())
    assert [s.name for s in p.targets] == ["Pallavi"]
    p = place_revisions(["reshape the ending"], _tune())
    assert [s.name for s in p.targets] == ["Outro"]


def test_a_point_on_a_boundary_belongs_to_the_section_that_starts_there():
    p = place_revisions(["the note at 16.6 s"], _tune())
    assert [s.name for s in p.targets] == ["Anupallavi"]
    p = place_revisions(["the note at 0 s"], _tune())
    assert [s.name for s in p.targets] == ["Prelude"]
    p = place_revisions(["the last note at 62.0 s"], _tune())
    assert [s.name for s in p.targets] == ["Outro"]


def test_a_range_ending_on_a_boundary_does_not_reach_the_next_section():
    p = place_revisions(["the passage at 5.5-16.6 s"], _tune())
    assert [s.name for s in p.targets] == ["Pallavi"]
    p = place_revisions(["from 16.6 to 25.0 s"], _tune())
    assert [s.name for s in p.targets] == ["Anupallavi"]
    p = place_revisions(["from 16.5 to 25.1 s"], _tune())
    assert [s.name for s in p.targets] == ["Pallavi", "Anupallavi", "Interlude 1"]
