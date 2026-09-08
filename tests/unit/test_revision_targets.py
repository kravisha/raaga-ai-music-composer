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


def test_a_locked_section_is_named_and_left_alone():
    tune = _tune()
    tune.sections[1].locked = True
    p = place_revisions(["retain the Pallavi's motif; develop the Charanam"], tune)
    assert [s.name for s in p.targets] == ["Charanam 1"]
    assert p.skipped_locked == ["Pallavi"]
    assert "Pallavi is locked" in " ".join(p.notes)


def test_a_revision_that_names_nothing_is_unplaced():
    p = place_revisions(["make it warmer", "more variation"], _tune())
    assert not p.targets and len(p.unplaced) == 2 and not p.placed_anything
    assert "name no passage" in " ".join(p.notes)


def test_targets_come_back_in_time_order_and_once():
    p = place_revisions(["fix the Outro", "the Prelude drags", "again the outro"], _tune())
    assert [s.name for s in p.targets] == ["Prelude", "Outro"]
