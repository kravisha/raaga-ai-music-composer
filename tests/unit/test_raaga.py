"""Unit tests: raaga knowledge store, selection engine and song structure."""
from __future__ import annotations

import json

import pytest

from raagacomposer.core.models import CreativeBrief, Section, SectionKind
from raagacomposer.music.structure import (choose_template, describe,
                                           plan_sections, section_role,
                                           sections_asked_for)
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


# --------------------------------------------------------------------------
# Selecting a raaga shows what is known about it (Arya, 2026-09-07 08:39:12)
# --------------------------------------------------------------------------
def test_a_curated_raaga_describes_what_the_specification_lists():
    """Krish selected Hamsadhwani and saw a fraction of what was known.

    The specification's per-raaga list names characteristic phrases (item 6)
    and gamaka behaviour (item 9), and elsewhere aliases and the
    melakarta/janya relation.  The library held all of it; describe() showed
    none of it.
    """
    from raagacomposer.raaga.library import library

    text = library().require("Hamsadhwani").describe()
    for expected in ("Arohanam", "Avarohanam", "Jeeva swaras",
                     "Resting (nyasa)", "Characteristic phrases", "Gamaka",
                     "Starts on", "Avoid"):
        assert expected in text, f"{expected!r} is missing from the details"

    # and the content, not merely the label
    assert "S  R2  G3  P" in text, "the prayogas are not actually shown"
    assert "kampita" in text, "the gamaka behaviour is not actually shown"
    assert "M1" in text.split("Avoid:")[1], "the avoided swaras are not shown"


def test_a_scale_only_raaga_does_not_claim_what_it_lacks():
    """Honest degradation: no invented phrases for a parent scale."""
    from raagacomposer.raaga.library import library

    scale_only = next(r for r in library().all() if r.scale_only)
    text = scale_only.describe()
    assert "Characteristic phrases" not in text
    assert "Gamaka:" not in text
    assert "parent scale" in text


def test_aliases_are_shown_when_the_library_has_them():
    """A raaga spelled three ways is one raaga, and saying so helps."""
    from raagacomposer.raaga.library import library

    with_aliases = next(r for r in library().all() if r.aliases)
    assert "Also called:" in with_aliases.describe()


def test_evidence_is_shown_with_its_basis_labelled():
    """Krish: information used to recommend a raaga must be visible in its
    details, even when it has not been verified through learning.

    Chitrambari was recommended on its block characters and tags, and the
    display showed an empty "moods" heading and nothing else - the one
    absent field made prominent while the evidence that drove the choice
    was invisible.
    """
    from raagacomposer.raaga.library import library

    text = library().require("Chitrambari").describe()
    assert "Moods (curated): none recorded" in text, \
        "an absent field must say so rather than show a blank heading"
    assert "Descriptors (reference pack, not studied)" in text
    assert "Scale character (derived from its blocks)" in text
    assert "luminous" in text, "the descriptors are labelled but not shown"


def test_reference_descriptors_are_never_presented_as_curated_moods():
    """The label carries the claim.  A descriptor inferred from a scale is
    not a mood a person wrote down, and merging them would be fabrication."""
    from raagacomposer.raaga.library import library

    raaga = library().require("Chitrambari")
    text = raaga.describe()
    moods_line = next(l for l in text.splitlines() if "Moods (curated)" in l)
    for tag in raaga.tags:
        assert tag not in moods_line, \
            f"reference tag {tag!r} was promoted into the curated moods line"


def test_the_comparison_shows_both_raagas_evidence():
    """Comparing a curated raaga with a pack entry used to give one
    populated column beside two blank headings."""
    from raagacomposer.raaga.library import library
    from raagacomposer.raaga.selection import compare

    lib = library()
    text = compare(lib.require("Keeravani"), lib.require("Chitrambari"))
    assert text.count("Descriptors (reference pack, not studied)") == 2
    assert "Moods (curated): romantic" in text          # the curated one
    assert "Moods (curated): none recorded" in text     # and the honest gap
    assert "" != text.split("Chitrambari:")[1].strip()


def test_a_section_the_creator_named_survives_a_short_song():
    """Arya's 60-second case.  The brief said "Include Prelude, Pallavi,
    Anupallavi, Interlude, Charanam and Ending" and the tune came back
    without an Anupallavi and without a word about it, because the planner
    saw a template and a duration and never saw the brief."""
    asked = sections_asked_for(
        "Include Prelude, Pallavi, Anupallavi, Interlude, Charanam "
        "and Ending.")
    assert SectionKind.ANUPALLAVI in asked

    unasked = plan_sections(60.0, 72, 8, "film song")
    assert not any(s.kind is SectionKind.ANUPALLAVI for s in unasked), \
        "this test proves nothing if the planner keeps it anyway"

    notes = []
    sections = plan_sections(60.0, 72, 8, "film song", requested=asked,
                             notes=notes)
    kinds = [s.kind for s in sections]
    for kind in asked:
        assert kind in kinds, f"{kind.value} was asked for and dropped"
    for a, b in zip(sections, sections[1:]):
        assert b.start == pytest.approx(a.end)

    # Sixty seconds is not actually a conflict once the repeats nobody
    # asked for are gone, so there is nothing to explain and the planner
    # should not invent a complaint.
    assert notes == [], notes
    assert 50 <= sections[-1].end <= 70, sections[-1].end

    # Asking for one interlude is not asking for both of the template's.
    assert sum(1 for k in kinds if k is SectionKind.INTERLUDE) == 1, \
        [s.name for s in sections]


def test_a_song_too_short_for_what_was_asked_for_says_so():
    """When it genuinely will not fit, the creator hears why rather than
    getting a song that quietly disagrees with the brief."""
    asked = sections_asked_for(
        "Include Prelude, Pallavi, Anupallavi, Interlude, Charanam "
        "and Ending.")
    notes = []
    sections = plan_sections(30.0, 72, 8, "film song", requested=asked,
                             notes=notes)
    for kind in asked:
        assert kind in [s.kind for s in sections], \
            f"{kind.value} was dropped instead of explained"
    assert notes, "the song came back shorter than asked with nothing said"
    said = " ".join(notes)
    assert "30s" in said and "kept every section you named" in said, said


def test_asking_for_a_pallavi_is_not_asking_for_an_anupallavi():
    """One name contains the other, and the spoken spelling puts a space
    in the middle of the longer one."""
    assert sections_asked_for("add a pallavi") == (SectionKind.PALLAVI,)
    assert sections_asked_for("add an anupallavi") == (SectionKind.ANUPALLAVI,)
    assert sections_asked_for("a hopeful, romantic song") == ()

    # "Anu Pallavi" is one section, not an Anupallavi and a Pallavi.
    spoken = sections_asked_for("include an Anu Pallavi")
    assert spoken == (SectionKind.ANUPALLAVI,), spoken

    listed = sections_asked_for(
        "Include Prelude, Pallavi, Anu Pallavi, Interlude, Charanam "
        "and Ending.")
    assert SectionKind.ANUPALLAVI in listed and SectionKind.PALLAVI in listed


def test_a_story_is_not_a_list_of_sections():
    """The docstring used to promise this and the code did the opposite:
    any occurrence of any section name anywhere became an instruction."""
    from raagacomposer.music.structure import read_section_requests

    for narrative in ("A bridge between two worlds",
                      "A happy ending to their long separation",
                      "the interlude of his life between two cities",
                      "a young novice musician eager to impress"):
        got = read_section_requests("", "", narrative)
        assert not got.wanted, f"{narrative!r} was read as a request: {got}"
        assert not got.refused, f"{narrative!r} was read as a refusal: {got}"

    # A name still counts when it is actually asked for, or listed.
    assert read_section_requests("include a bridge").wanted == \
        (SectionKind.BRIDGE,)
    listed = read_section_requests("pallavi, charanam, outro").wanted
    assert SectionKind.PALLAVI in listed and SectionKind.OUTRO in listed


def test_a_section_the_creator_ruled_out_stays_out():
    """A list of what was wanted cannot say what was not: "do not include
    an Anupallavi" is not silence about the Anupallavi."""
    from raagacomposer.music.structure import read_section_requests

    asked = read_section_requests(
        "Do not include Anupallavi; include Pallavi and Charanam.")
    assert SectionKind.ANUPALLAVI in asked.refused
    assert SectionKind.ANUPALLAVI not in asked.wanted
    assert SectionKind.PALLAVI in asked.wanted

    sections = plan_sections(150.0, 72, 8, "film song",
                             requested=asked.wanted, refused=asked.refused)
    assert not any(s.kind is SectionKind.ANUPALLAVI for s in sections), \
        [s.name for s in sections]
    assert any(s.kind is SectionKind.PALLAVI for s in sections)
    for a, b in zip(sections, sections[1:]):
        assert b.start == pytest.approx(a.end)


def test_reprises_nobody_asked_for_go_before_the_apology():
    """Arya's 45-second case.  Six requested kinds need 40s at one cycle
    each, and the template's unrequested Pallavi 2 and Pallavi 3 forced
    53s - so the planner apologised for a conflict it had created."""
    from raagacomposer.music.structure import read_section_requests

    asked = read_section_requests(
        "Include Prelude, Pallavi, Anupallavi, Interlude, Charanam "
        "and Ending.")
    notes = []
    sections = plan_sections(45.0, 72, 8, "film song",
                             requested=asked.wanted, refused=asked.refused,
                             notes=notes)
    assert notes == [], notes
    assert sections[-1].end <= 45.0 * 1.15, sections[-1].end
    for kind in asked.wanted:
        assert kind in [s.kind for s in sections], kind


def test_a_kept_reprise_is_not_numbered_around_a_missing_one():
    """Dropping Pallavi 2 and keeping Pallavi 3 reads as a missing
    section rather than a shorter song."""
    from raagacomposer.music.structure import read_section_requests

    asked = read_section_requests(
        "Include Prelude, Pallavi, Anupallavi, Interlude, Charanam "
        "and Ending.")
    sections = plan_sections(150.0, 72, 8, "film song",
                             requested=asked.wanted, refused=asked.refused)
    names = [s.name for s in sections]
    assert "Pallavi 3" not in names or "Pallavi 2" in names, names


def test_a_section_the_template_does_not_have_is_added_when_asked_for():
    plain = plan_sections(150.0, 72, 8, "devotional")
    assert not any(s.kind is SectionKind.BRIDGE for s in plain)

    notes = []
    with_bridge = plan_sections(150.0, 72, 8, "devotional",
                                requested=(SectionKind.BRIDGE,), notes=notes)
    kinds = [s.kind for s in with_bridge]
    assert SectionKind.BRIDGE in kinds
    assert kinds[-1] is SectionKind.OUTRO, "the bridge landed after the ending"
    assert any("Bridge" in n for n in notes), notes


def test_a_long_song_still_drops_what_nobody_asked_for():
    """Keeping a named section must not turn into keeping everything."""
    everything = plan_sections(240.0, 72, 8, "film song")
    short = plan_sections(60.0, 72, 8, "film song",
                          requested=(SectionKind.ANUPALLAVI,))
    assert len(short) < len(everything), \
        "the short song kept as much as the long one"
    assert any(s.kind is SectionKind.ANUPALLAVI for s in short)
