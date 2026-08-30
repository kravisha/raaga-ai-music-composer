"""Unit tests: pitch/time helpers and the project data model."""
from __future__ import annotations

import pytest

from raagacomposer.core.models import (ApprovalState, ArrangementVersion,
                                       CreativeBrief, LyricLine, LyricsVersion,
                                       MelodyVersion, MixVersion, Note, Project,
                                       Region, Section, SectionKind, Stage,
                                       Track, VocalRender)
from raagacomposer.core.serde import from_jsonable, to_jsonable
from raagacomposer.music import theory

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# theory
# --------------------------------------------------------------------------
def test_midi_and_frequency_round_trip():
    assert theory.midi_to_freq(69) == pytest.approx(440.0)
    assert theory.midi_to_freq(81) == pytest.approx(880.0)
    assert theory.freq_to_midi(440.0) == pytest.approx(69.0)
    assert theory.midi_name(60) == "C4"


def test_beat_and_cycle_lengths():
    assert theory.beat_seconds(60) == pytest.approx(1.0)
    assert theory.beat_seconds(120) == pytest.approx(0.5)
    assert theory.cycle_seconds(60, 8) == pytest.approx(8.0)


def test_fit_to_range_moves_by_octaves_only():
    fitted = theory.fit_to_range(40, 55, 79)
    assert fitted == 64
    assert 55 <= fitted <= 79
    assert (fitted - 40) % 12 == 0
    assert theory.fit_to_range(96, 55, 79) == 72


def test_quantise_snaps_to_the_grid():
    assert theory.quantise(0.26, 120, 0.25) == pytest.approx(0.25)
    assert theory.quantise(0.60, 120, 0.5) == pytest.approx(0.5)


def test_time_formatting():
    assert theory.format_time(0) == "0:00.00"
    assert theory.format_time(65.5) == "1:05.50"
    assert theory.format_time_short(125) == "2:05"


def test_spread_scales_weights_to_a_total():
    out = theory.spread([1, 1, 2], 8.0)
    assert sum(out) == pytest.approx(8.0)
    assert out[2] == pytest.approx(4.0)


# --------------------------------------------------------------------------
# model behaviour
# --------------------------------------------------------------------------
def test_note_and_section_geometry():
    note = Note(start=2.0, duration=0.5)
    assert note.end == pytest.approx(2.5)
    section = Section(start=10.0, end=20.0)
    assert section.duration == pytest.approx(10.0)
    assert section.contains(10.0) and section.contains(19.9)
    assert not section.contains(20.0)


def test_section_kind_knows_which_are_instrumental():
    assert SectionKind.PRELUDE.instrumental
    assert SectionKind.INTERLUDE.instrumental
    assert not SectionKind.PALLAVI.instrumental
    assert not SectionKind.CHARANAM.instrumental


def test_region_overlap_is_half_open():
    region = Region(start=10.0, end=20.0)
    assert region.overlaps(19.9, 25.0)
    assert not region.overlaps(20.0, 25.0)
    assert not region.overlaps(0.0, 10.0)


def test_track_span_and_lookup():
    a = Region(start=5.0, end=10.0)
    b = Region(start=20.0, end=30.0)
    track = Track(instrument="veena", regions=[a, b])
    assert track.start == pytest.approx(5.0)
    assert track.end == pytest.approx(30.0)
    assert track.region_by_id(b.id) is b
    assert track.regions_in(9.0, 21.0) == [a, b]
    assert track.label == "Veena"


def test_project_accessors_pick_the_approved_versions():
    project = Project(title="T")
    m1 = MelodyVersion(version=1, notes=[Note(start=0, duration=1)])
    m2 = MelodyVersion(version=2, notes=[Note(start=0, duration=2)])
    project.melodies = [m1, m2]
    assert project.melody() is m2               # latest when none approved
    project.approved_melody = 1
    assert project.melody() is m1
    assert project.melody(2) is m2

    lv = LyricsVersion(version=1, lines=[LyricLine(text="a")])
    project.lyrics = [lv]
    assert project.lyrics_version() is lv

    arrangement = ArrangementVersion(version=1)
    project.arrangements = [arrangement]
    assert project.arrangement() is arrangement


def test_locked_melody_only_reports_accepted_tunes():
    project = Project()
    melody = MelodyVersion(version=1, notes=[Note()])
    project.melodies = [melody]
    project.approved_melody = 1
    assert project.locked_melody is None
    melody.state = ApprovalState.LOCKED
    assert project.locked_melody is melody


def test_vocal_master_prefers_the_marked_take():
    project = Project()
    preview = VocalRender(version=1, kind="preview")
    master = VocalRender(version=2, kind="master")
    project.vocal_renders = [preview, master]
    assert project.vocal_master is master
    project.vocal_master_id = master.id
    assert project.vocal_master is master
    assert project.latest_vocal is master


def test_project_duration_falls_back_to_the_brief():
    project = Project()
    project.brief.duration_target = 150.0
    assert project.duration == pytest.approx(150.0)
    project.melodies = [MelodyVersion(version=1,
                                      notes=[Note(start=0.0, duration=42.0)])]
    project.approved_melody = 1
    assert project.duration == pytest.approx(42.0)
    project.mixes = [MixVersion(version=1, duration=61.0)]
    assert project.duration == pytest.approx(61.0)


def test_history_is_capped():
    project = Project()
    for i in range(2100):
        project.log_history("test", f"entry {i}")
    assert len(project.history) == 2000
    assert project.history[-1].description == "entry 2099"


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------
def test_full_project_round_trips_through_json():
    project = Project(title="Round Trip")
    project.current_stage = Stage.ARRANGEMENT
    project.brief = CreativeBrief(mood="longing", language="Tamil",
                                  instruments_preferred=["veena"])
    section = Section(name="Pallavi", kind=SectionKind.PALLAVI, start=0, end=8)
    melody = MelodyVersion(version=1, raaga="Keeravani", sections=[section],
                           notes=[Note(swara="G2", midi=63, start=0.0,
                                       duration=0.5, section_id=section.id)],
                           state=ApprovalState.LOCKED)
    project.melodies = [melody]
    project.approved_melody = 1
    track = Track(instrument="veena", regions=[Region(start=0, end=8,
                                                      notes=[Note()])])
    project.arrangements = [ArrangementVersion(version=1, tracks=[track])]

    clone = from_jsonable(Project, to_jsonable(project))

    assert clone.title == "Round Trip"
    assert clone.current_stage is Stage.ARRANGEMENT
    assert clone.brief.instruments_preferred == ["veena"]
    assert clone.melody().state is ApprovalState.LOCKED
    assert clone.melody().sections[0].kind is SectionKind.PALLAVI
    assert clone.melody().notes[0].swara == "G2"
    assert clone.arrangement().tracks[0].regions[0].notes


def test_serde_tolerates_missing_and_unknown_fields():
    data = to_jsonable(Project(title="Partial"))
    data.pop("mixes", None)
    data["something_new"] = 42
    clone = from_jsonable(Project, data)
    assert clone.title == "Partial"
    assert clone.mixes == []


def test_serde_falls_back_for_an_unknown_enum_value():
    data = to_jsonable(Project())
    data["current_stage"] = "not-a-stage"
    clone = from_jsonable(Project, data)
    assert isinstance(clone.current_stage, Stage)
