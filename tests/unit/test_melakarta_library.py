"""The 72 parent scales inside the library - docs/PLAN_stage1_knowledge.md S1.

``tests/unit/test_stage1_pack.py`` checks the pack as a document.  This file
checks what the application does with it: that all 72 melakartas are loaded,
that the eight curated entries keep everything they had and gain only what
the pack knows, that a melakarta nobody curated is honest about being a scale
and still composes real music from it, and that seventy-two names in one
namespace do not start answering for each other.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.agent.practice import drift_neighbours
from raagacomposer.core.models import CreativeBrief
from raagacomposer.music.melody import MelodyOptions, generate
from raagacomposer.raaga.library import RaagaLibrary
from raagacomposer.raaga.selection import suggest

pytestmark = pytest.mark.unit

#: A path that cannot exist, so a creator's own raagas_user.json can never
#: change what these tests see.
NO_USER_FILE = Path("no-such-user-raagas.json")


@pytest.fixture(scope="module")
def lib() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE)


@pytest.fixture(scope="module")
def curated_only() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE, melakartas=False)


# -- the 72 are there -------------------------------------------------------
def test_the_library_carries_all_seventy_two_melakartas(lib):
    numbers = sorted(r.melakarta for r in lib.all() if r.melakarta)
    assert numbers == list(range(1, 73))


def test_the_janya_raagas_are_still_there(lib, curated_only):
    """Loading the pack adds; it never removes what was curated."""
    for raaga in curated_only.all():
        assert lib.get(raaga.name) is not None, raaga.name
    assert len(lib.all()) == len(curated_only.all()) + 64


def test_pack_test_c_endpoints_are_live_library_entries(lib):
    """Pack document 06 test C, against the library rather than the files."""
    for number, name, arohanam in (
            (1, "Kanakangi", "S R1 G1 M1 P D1 N1 S+"),
            (36, "Chalanata", "S R3 G3 M1 P D3 N3 S+"),
            (37, "Salagam", "S R1 G1 M2 P D1 N1 S+"),
            (72, "Rasikapriya", "S R3 G3 M2 P D3 N3 S+")):
        raaga = lib.require(name)
        assert raaga.melakarta == number
        assert " ".join(raaga.arohanam) == arohanam
        assert list(reversed(raaga.arohanam)) == raaga.avarohanam


def test_every_melakarta_knows_which_blocks_it_is_made_of(lib):
    for raaga in lib.all():
        if not raaga.melakarta:
            continue
        assert raaga.rg and raaga.madhyama and raaga.dn, raaga.name
        assert raaga.madhyama == ("M1" if raaga.melakarta <= 36 else "M2")
        # The blocks are the scale, not a label attached to a name.
        bases = [s.rstrip("+-") for s in raaga.arohanam]
        assert raaga.rg == bases[1] + bases[2], raaga.name
        assert raaga.dn == bases[5] + bases[6], raaga.name
        assert raaga.block_summary().startswith(raaga.rg)


# -- the curated entries win ------------------------------------------------
def test_a_curated_melakarta_keeps_everything_it_had(lib, curated_only):
    """The pack has no prayogas, nyasa, gamaka or tempo; it must take none away."""
    for name in ("Keeravani", "Kalyani", "Shankarabharanam", "Kharaharapriya",
                 "Hanumatodi", "Natabhairavi", "Charukesi", "Mayamalavagowla"):
        before, after = curated_only.require(name), lib.require(name)
        assert after.prayogas == before.prayogas, name
        assert after.jeeva == before.jeeva and after.nyasa == before.nyasa, name
        assert after.gamaka == before.gamaka, name
        assert after.moods == before.moods, name
        assert after.tempo_range == before.tempo_range, name
        assert after.notes == before.notes, name
        assert not after.scale_only, name


def test_a_curated_melakarta_gains_what_the_pack_knows(lib):
    keeravani = lib.require("Keeravani")
    assert keeravani.chakra == "4-Veda"
    assert (keeravani.rg, keeravani.madhyama, keeravani.dn) == ("R2G2", "M1", "D1N3")
    assert "tender" in keeravani.tags
    assert "poignant" in keeravani.block_summary()


def test_the_packs_own_spelling_reaches_the_curated_entry(lib):
    """One melakarta is one entry, whichever name the creator types."""
    for pack_name, curated in (("Mechakalyani", "Kalyani"),
                               ("Dheerasankarabharanam", "Shankarabharanam")):
        assert lib.get(pack_name) is lib.require(curated)
        assert lib.get(pack_name).melakarta == lib.require(curated).melakarta


# -- a melakarta nobody curated --------------------------------------------
def test_a_scale_only_melakarta_says_that_is_all_it_is(lib):
    raaga = lib.require("Simhendramadhyamam")
    assert raaga.melakarta == 57
    assert raaga.scale_only
    assert not raaga.prayogas and not raaga.jeeva and not raaga.nyasa
    assert not raaga.gamaka and not raaga.moods
    # No tempo range invented for it either: section 37, unknown stays unknown.
    assert raaga.tempo_range == []
    assert "parent scale" in raaga.character()
    assert "R2G2 tender" in raaga.character()


def test_a_scale_only_melakarta_composes_real_music_from_its_scale(lib):
    """Specification TEST G, for a raaga the library knows only as a scale."""
    raaga = lib.require("Simhendramadhyamam")
    melody = generate(raaga, MelodyOptions(seed=5, tempo_bpm=72,
                                           duration_target=45))
    assert len(melody.notes) > 20
    assert len({n.midi for n in melody.notes}) > 3, "a tune, not one held note"
    allowed = set(raaga.allowed)
    outside = {n.swara for n in melody.notes
               if n.swara.rstrip("+-") not in allowed}
    assert not outside, f"{raaga.name} does not use {sorted(outside)}"


# -- seventy-two names in one namespace ------------------------------------
def test_a_janya_and_the_melakarta_named_after_it_stay_apart(lib):
    """The trap this whole merge is arranged to avoid."""
    bhairavi, natabhairavi = lib.require("Bhairavi"), lib.require("Natabhairavi")
    assert bhairavi is not natabhairavi
    assert bhairavi.melakarta is None and natabhairavi.melakarta == 20
    assert lib.find_in_text("something in natabhairavi") is natabhairavi
    assert lib.find_in_text("something in bhairavi") is bhairavi


def test_a_name_buried_inside_a_word_is_not_a_request_for_that_raaga(lib):
    """Whole words only, or a 72-name library starts hearing things."""
    assert lib.find_in_text("a pavanish sort of evening") is None
    assert lib.find_in_text("play something in pavani") is lib.require("Pavani")


def test_the_longest_name_in_the_text_wins(lib):
    assert lib.find_in_text("make it mechakalyani") is lib.require("Kalyani")
    assert lib.find_in_text("kalyani, please") is lib.require("Kalyani")


def test_no_two_raagas_answer_to_one_spelling(lib):
    """Every alias resolves to exactly one entry, and to an entry that exists."""
    seen = {}
    for raaga in lib.all():
        for name in [raaga.name] + list(raaga.aliases):
            key = name.lower()
            assert lib.get(name) is not None, name
            if key in seen and seen[key] != raaga.name:
                # A collision must not be silent; the library keeps the first
                # claim and logs, and the alias must still resolve to it.
                assert lib.get(name).name == seen[key], (name, seen[key])
            seen.setdefault(key, raaga.name)


# -- what this does to the drift exercises ---------------------------------
def test_the_confusable_neighbours_are_still_the_curated_raagas(lib):
    """The seam the pack opened: 72 scales tie on overlap and crowd out the
    raagas a student is actually taught to tell apart."""
    for name, expected in (("Keeravani", {"Bhairavi", "Natabhairavi",
                                          "Sindhu Bhairavi"}),
                           ("Shankarabharanam", {"Kalyani", "Kambhoji"}),
                           ("Mohanam", {"Kalyani", "Kambhoji",
                                        "Shankarabharanam"})):
        chosen = {r.name for r in drift_neighbours(lib, lib.require(name))}
        assert expected <= chosen, (name, sorted(chosen))


def test_a_bare_scale_never_displaces_a_curated_raaga_it_ties_with(lib):
    for raaga in (lib.require("Keeravani"), lib.require("Kalyani"),
                  lib.require("Mohanam"), lib.require("Bhairavi")):
        allowed = set(raaga.allowed)

        def overlap(other):
            return len(set(other.allowed) & allowed)

        chosen = drift_neighbours(lib, raaga)
        for scale in (r for r in chosen if r.scale_only):
            tied_and_curated = [r for r in lib.all()
                                if not r.scale_only and r.name != raaga.name
                                and overlap(r) == overlap(scale)]
            missing = [r.name for r in tied_and_curated if r not in chosen]
            assert not missing, (raaga.name, scale.name, missing)


# -- what this does to Apply Brief -----------------------------------------
def test_a_brief_that_says_nothing_still_gets_a_curated_answer(lib):
    """With no emotion to match, fall back to something we know about
    rather than to a parent scale nobody could give a reason for."""
    suggestions = suggest(CreativeBrief(mood="", feel="", situation=""),
                          lib, limit=5)
    assert suggestions
    assert not lib.require(suggestions[0].name).scale_only


def test_a_melakarta_nobody_curated_can_now_earn_its_place(lib, curated_only):
    """What S2 changed on purpose.

    Under S1 the 64 scale-only melakartas were carried but could not be
    ranked: they had no curated moods and mood matching was the whole of the
    score.  The block-character engine can speak for them, so they compete -
    and the reason they are offered with is traceable to their blocks.
    """
    brief = CreativeBrief(situation="love failure", mood="sad",
                          feel="lonely late at night but still warm")
    before = {s.name for s in suggest(brief, curated_only, limit=6)}
    after = [s for s in suggest(brief, lib, limit=6)]
    assert {s.name for s in after} != before, "the pack should change the answer"
    for suggestion in after:
        raaga = lib.require(suggestion.name)
        if raaga.scale_only:
            assert raaga.rg in suggestion.rationale or \
                raaga.dn in suggestion.rationale, suggestion.rationale


def test_a_scale_only_raaga_can_still_be_asked_for_by_name(lib):
    brief = CreativeBrief(situation="a temple at dawn", mood="devotional",
                          feel="I want this one in Shubhapantuvarali")
    top = suggest(brief, lib, limit=4)[0]
    assert top.name == "Shubhapantuvarali"
    assert "parent scale" in top.rationale
