"""Unit tests: raaga knowledge store, selection engine and song structure."""
from __future__ import annotations

import json

import pytest

from raagacomposer.core.models import CreativeBrief, Section, SectionKind
from raagacomposer.music.structure import (choose_template, describe,
                                           plan_sections, section_role)
from raagacomposer.raaga.library import (RaagaLibrary, parse_swara, swara_midi,
                                         swara_semitone)
from raagacomposer.raaga.selection import (compare, expand_feel_words,
                                           infer_tempo, suggest)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# swara notation
# --------------------------------------------------------------------------
def test_swara_octave_marks():
    assert parse_swara("S") == ("S", 0)
    assert parse_swara("S+") == ("S", 1)
    assert parse_swara("P-") == ("P", -1)
    assert parse_swara("G3++") == ("G3", 2)


def test_swara_semitones_and_midi(raagas):
    assert swara_semitone("S") == 0
    assert swara_semitone("P") == 7
    assert swara_semitone("S+") == 12
    assert swara_semitone("N3-") == -1
    assert swara_midi("P", 60) == 67
    assert swara_midi("S+", 60) == 72


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------
def test_library_loads_the_shipped_set(raagas: RaagaLibrary):
    names = raagas.names()
    assert len(names) >= 15
    for expected in ("Kalyani", "Keeravani", "Mohanam", "Shankarabharanam"):
        assert expected in names


def test_lookup_by_name_and_alias(raagas: RaagaLibrary):
    assert raagas.get("kalyani").name == "Kalyani"
    assert raagas.get("Yaman").name == "Kalyani"       # Hindustani alias
    assert raagas.get("Bhoop").name == "Mohanam"
    assert raagas.get("no such raaga at all xyz") is None


def test_find_in_text_picks_the_longest_match(raagas: RaagaLibrary):
    found = raagas.find_in_text("please use raaga Mechakalyani for this one")
    assert found is not None and found.name == "Kalyani"
    assert raagas.find_in_text("play the first minute") is None


def test_asymmetric_raaga_keeps_its_ascending_and_descending_sets(raagas):
    abheri = raagas.require("Abheri")
    assert "R2" not in abheri.ascending
    assert "D2" not in abheri.ascending
    assert "R2" in abheri.descending and "D2" in abheri.descending
    # Stepping up from S must skip the notes the arohanam does not use.
    assert abheri.step("S", 1, 1) == "G2"


def test_pentatonic_raaga_reports_its_missing_notes(raagas):
    mohanam = raagas.require("Mohanam")
    assert "M1" not in mohanam.allowed
    assert set(mohanam.forbidden_swaras) >= {"M1", "N2", "N3"}


def test_step_degree_and_from_degree_are_consistent(keeravani):
    assert keeravani.degree("S") == 0
    assert keeravani.degree("S+") == 7
    assert keeravani.from_degree(7) == "S+"
    assert keeravani.step("S", 7, 1) == "S+"
    assert keeravani.step("S+", -7, -1) == "S"


def test_pitches_in_range_and_nearest_token(keeravani):
    pitches = keeravani.pitches_in_range(60, 60, 72)
    assert pitches[0] == 60 and pitches[-1] == 72
    assert all(60 <= p <= 72 for p in pitches)
    assert keeravani.nearest_token(67, 60) == "P"
    assert keeravani.nearest_token(72, 60) == "S+"


def test_gamaka_and_description(keeravani):
    assert keeravani.gamaka_for("G2")
    text = keeravani.describe()
    assert "Arohanam" in text and "Keeravani" in text


def test_user_raaga_file_extends_the_library(tmp_path):
    extra = tmp_path / "raagas_user.json"
    extra.write_text(json.dumps({
        "raagas": [{
            "name": "Test Raagam",
            "arohanam": ["S", "R2", "M1", "P", "N2", "S+"],
            "avarohanam": ["S+", "N2", "P", "M1", "R2", "S"],
            "jeeva": ["M1"], "nyasa": ["S", "P"], "graha": ["S"],
            "prayogas": [["S", "R2", "M1"]], "gamaka": {"M1": "kampita"},
            "moods": ["test"], "tempo_range": [60, 90],
        }]
    }), encoding="utf-8")
    lib = RaagaLibrary(extra_path=extra)
    added = lib.get("Test Raagam")
    assert added is not None
    assert added.source == "user"
    assert "M1" in added.allowed


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def test_expand_feel_words_reads_ordinary_language():
    words = expand_feel_words("lonely, late at night, but still warm")
    assert "lonely" in words
    assert "night" in words
    assert "warm" in words


def test_suggestions_match_a_sad_night_brief():
    brief = CreativeBrief(mood="longing",
                          feel="lonely, late at night, but still warm")
    names = [s.name for s in suggest(brief)]
    assert names
    assert any(n in names for n in ("Shivaranjani", "Keeravani", "Charukesi"))


def test_suggestions_match_a_celebration_brief():
    brief = CreativeBrief(mood="celebration", feel="festive wedding, bright")
    names = [s.name for s in suggest(brief)]
    assert any(n in names for n in ("Hamsadhwani", "Kalyani", "Mohanam",
                                    "Shankarabharanam"))


def test_an_explicit_request_wins():
    brief = CreativeBrief(mood="sad", raaga_preference="Kalyani")
    top = suggest(brief)[0]
    assert top.name == "Kalyani"
    assert "asked for" in top.rationale.lower()


def test_suggestion_always_returns_something():
    assert suggest(CreativeBrief(mood="", feel="", situation=""))


def test_infer_tempo_respects_preference_and_feel(keeravani):
    assert infer_tempo(CreativeBrief(tempo_preference=96), keeravani) == 96
    slow = infer_tempo(CreativeBrief(mood="sad", feel="very slow"), keeravani)
    fast = infer_tempo(CreativeBrief(mood="celebration", feel="fast"), keeravani)
    assert slow < fast


def test_compare_reports_the_differing_notes(raagas):
    text = compare(raagas.require("Mohanam"), raagas.require("Hamsadhwani"))
    assert "Mohanam" in text and "Hamsadhwani" in text
    assert "Only in" in text


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------
def test_sections_add_up_to_the_requested_length():
    sections = plan_sections(150.0, 72, 8, "film song")
    assert sections
    total = sections[-1].end
    assert 120 <= total <= 185
    for a, b in zip(sections, sections[1:]):
        assert b.start == pytest.approx(a.end)


def test_short_songs_drop_the_optional_sections():
    long_song = plan_sections(240.0, 72, 8, "film song")
    short_song = plan_sections(60.0, 72, 8, "film song")
    assert len(short_song) <= len(long_song)
    assert short_song[0].kind is SectionKind.PRELUDE
    assert short_song[-1].kind is SectionKind.OUTRO


def test_named_sections_the_creator_can_ask_for_exist():
    names = {s.name.lower() for s in plan_sections(180.0, 72, 8, "film song")}
    assert "prelude" in names
    assert any("pallavi" in n for n in names)
    assert any("interlude" in n for n in names)
    assert any("charanam" in n for n in names)
    assert "outro" in names


def test_locked_sections_keep_their_length_when_replanned():
    original = plan_sections(150.0, 72, 8, "film song")
    original[1].locked = True
    kept_duration = original[1].duration
    replanned = plan_sections(200.0, 72, 8, "film song", existing=original)
    match = next(s for s in replanned if s.name == original[1].name)
    assert match.locked
    assert match.duration == pytest.approx(kept_duration)
    assert match.id == original[1].id


def test_templates_and_roles():
    assert choose_template("devotional") is not None
    assert section_role(SectionKind.PALLAVI) == "hook"
    assert section_role(SectionKind.CHARANAM) == "verse"
    assert section_role(SectionKind.INTERLUDE) == "instrumental"
    assert "Prelude" in describe(plan_sections(120.0, 72, 8, "film song"))
