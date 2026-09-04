"""The brief-to-raaga engine - Stage 1 pack document 05, plan item S2.

The pack's claim is that a raaga can be suggested for a reason a person can
check: its R-G block, its madhyama and its D-N block against fourteen emotion
dimensions read out of the brief.  These tests hold the engine to that -
including the pack's own mandatory test D, by name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.raaga import emotion
from raagacomposer.raaga.emotion import (DIMENSIONS, EmotionVector,
                                         profile_vector, rank, read_text,
                                         score_raaga, spread, target_vector)
from raagacomposer.raaga.library import RaagaLibrary

pytestmark = pytest.mark.unit

NO_USER_FILE = Path("no-such-user-raagas.json")


@pytest.fixture(scope="module")
def lib() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE)


# -- the vocabulary ---------------------------------------------------------
def test_the_fourteen_dimensions_are_the_packs_own():
    assert len(DIMENSIONS) == 14 == len(set(DIMENSIONS))
    for name in ("sadness", "tenderness", "yearning", "romance", "devotion",
                 "serenity", "joy", "warmth", "brightness", "gravity",
                 "mystery", "tension", "power", "wonder"):
        assert name in DIMENSIONS


def test_every_lexicon_entry_names_real_dimensions():
    for word, vector in list(emotion.LEXICON.items()) + list(emotion.PHRASES.items()):
        for dimension, weight in vector.items():
            assert dimension in DIMENSIONS, (word, dimension)
            assert 0.0 <= weight <= 1.0, (word, dimension, weight)


def test_the_block_characters_are_all_readable():
    """Every word the pack uses to describe a block has to mean something
    here, or a melakarta's profile is built from silence."""
    unreadable = []
    for block in ("compressed, austere, tense, vivadi-colored",
                  "plaintive, inward, grave",
                  "wide-contrast, dignified, austere-bright",
                  "tender, introspective, humane",
                  "open, lyrical, confident",
                  "bright-edged, urgent, intense",
                  "grounded, earthy, settled",
                  "luminous, searching, mysterious, heightened",
                  "compressed, dark, tense, unresolved",
                  "soft-dark, plaintive, descending pathos",
                  "poignant contrast, dramatic upward pull, strong resolution",
                  "warm, gentle, rounded, relaxed",
                  "open, affirmative, expansive, strong resolution"):
        if read_text(block).empty:
            unreadable.append(block)
    assert not unreadable


# -- reading a brief --------------------------------------------------------
def test_a_word_lands_in_the_dimension_it_means():
    assert read_text("sad")["sadness"] > 0.8
    assert read_text("joyful")["joy"] > 0.8
    assert read_text("mysterious")["mystery"] > 0.8


def test_negation_suppresses_rather_than_inverts():
    """"not sad" says sadness is not wanted; it does not say what is."""
    plain, negated = read_text("sad and tired"), read_text("not sad, just tired")
    assert plain["sadness"] > 0.5
    assert negated["sadness"] == 0.0


def test_intensity_words_scale_what_follows():
    assert read_text("very sad")["sadness"] >= read_text("sad")["sadness"]
    assert read_text("slightly sad")["sadness"] < read_text("sad")["sadness"]


def test_a_brief_with_only_a_mood_is_still_read():
    """Field weights renormalise over the fields that say something."""
    vector = target_vector(CreativeBrief(mood="devotional", feel="",
                                         situation=""))
    assert not vector.empty
    assert vector["devotion"] > 0.8


def test_the_packs_worked_example_reads_as_the_pack_describes_it():
    """Document 05 section 1's own example brief."""
    vector = target_vector(CreativeBrief(
        situation="love failure", mood="sad romantic",
        feel="lonely late at night but still warm"))
    for wanted in ("sadness", "yearning", "romance", "warmth", "tenderness"):
        assert vector[wanted] > 0.3, wanted
    for unwanted in ("joy", "brightness", "power"):
        assert vector[unwanted] < 0.2, unwanted


def test_an_empty_brief_reads_as_nothing():
    assert target_vector(CreativeBrief(mood="", feel="", situation="")).empty


# -- reading a raaga --------------------------------------------------------
def test_a_scale_only_melakarta_still_has_a_profile(lib):
    """The point of the block model: a raaga nobody curated can be spoken for."""
    vector = profile_vector(lib.require("Shubhapantuvarali"))
    assert not vector.empty
    assert vector["sadness"] > 0.3 or vector["yearning"] > 0.3


def test_a_curated_raaga_is_described_by_both_halves(lib):
    keeravani = profile_vector(lib.require("Keeravani"))
    assert keeravani["tenderness"] > 0.3       # from R2G2
    assert keeravani["romance"] > 0.3          # from its curated moods


def test_similarity_is_a_shape_not_a_loudness():
    one = EmotionVector({"sadness": 1.0, "warmth": 0.5})
    twice = EmotionVector({"sadness": 0.5, "warmth": 0.25})
    assert one.similarity(twice) == pytest.approx(1.0)
    assert one.similarity(EmotionVector({})) == 0.0


# -- the pack's penalties and bonuses --------------------------------------
def test_a_sad_brief_is_penalised_for_an_urgent_bright_edged_profile(lib):
    brief = CreativeBrief(mood="sad", feel="gentle and tender")
    urgent = score_raaga(brief, lib.require("Varunapriya"))    # R2G2 / D3N3
    assert urgent.penalties, "D3N3 against a tender brief should be objected to"


def test_a_peaceful_brief_is_penalised_for_an_unresolved_profile(lib):
    brief = CreativeBrief(mood="calm", feel="peaceful and still")
    tense = score_raaga(brief, lib.require("Jhankaradhwani"))   # D1N1
    assert any("unresolved" in p for p in tense.penalties)


def test_a_celebratory_brief_is_penalised_for_a_grave_profile(lib):
    brief = CreativeBrief(mood="joyful", feel="bright festive celebration")
    grave = score_raaga(brief, lib.require("Hanumatodi"))       # R1G2 / D1N2
    assert grave.penalties


def test_a_mysterious_brief_prefers_the_prati_madhyama(lib):
    brief = CreativeBrief(mood="mysterious", feel="strange and searching")
    m2 = score_raaga(brief, lib.require("Shubhapantuvarali"))   # M2
    assert any("M2" in b for b in m2.bonuses)


def test_a_warm_romantic_brief_prefers_the_warm_blocks(lib):
    brief = CreativeBrief(mood="romantic", feel="warm and gentle")
    warm = score_raaga(brief, lib.require("Kharaharapriya"))    # R2G2 / D2N2
    assert warm.bonuses


def test_bonuses_and_penalties_are_bounded(lib):
    """They tune the ranking; the fit is what decides it."""
    for name in ("Keeravani", "Kalyani", "Chalanata", "Shubhapantuvarali"):
        raaga = lib.require(name)
        for brief in (CreativeBrief(mood="sad", feel="tender and lonely"),
                      CreativeBrief(mood="joyful", feel="bright celebration"),
                      CreativeBrief(mood="mysterious", feel="devotional intensity")):
            target = target_vector(brief)
            adjustment, _, _ = emotion._adjustments(target, raaga)
            assert -emotion.MAX_PENALTY <= adjustment <= emotion.MAX_BONUS


def test_the_score_does_not_saturate(lib):
    """Clamping every good answer to 100 threw away the ranking."""
    brief = CreativeBrief(situation="love failure", mood="sad romantic",
                          feel="lonely late at night but still warm")
    scores = [s.score for s in rank(brief, lib.all(), limit=5)]
    assert max(scores) < 100.0
    assert len(set(scores)) > 1


# -- the spread -------------------------------------------------------------
def test_the_spread_keeps_the_best_first_and_the_scores_descending(lib):
    brief = CreativeBrief(mood="sad", feel="lonely and warm")
    chosen = rank(brief, lib.all(), limit=5)
    scores = [s.score for s in chosen]
    assert scores == sorted(scores, reverse=True)
    everything = [score_raaga(brief, r) for r in lib.all()]
    assert chosen[0].score == max(s.score for s in everything)


def test_the_spread_is_not_five_of_the_same_raaga(lib):
    """Pack document 05 section 5: avoid five almost-identical profiles."""
    brief = CreativeBrief(situation="love failure", mood="sad romantic",
                          feel="lonely late at night but still warm")
    chosen = rank(brief, lib.all(), limit=5)
    signatures = {(s.raaga.rg, s.raaga.madhyama, s.raaga.dn) for s in chosen}
    assert len(signatures) >= 4, [s.name for s in chosen]


def test_every_alternative_says_what_it_offers(lib):
    brief = CreativeBrief(mood="sad", feel="lonely late at night but warm")
    chosen = rank(brief, lib.all(), limit=5)
    assert chosen[0].role == "the closest fit"
    roles = [s.role for s in chosen]
    assert len(roles) == len(set(roles)), roles


def test_diversity_never_promotes_a_worse_raaga_to_the_top(lib):
    brief = CreativeBrief(mood="joyful", feel="bright wedding celebration")
    plain = sorted((score_raaga(brief, r) for r in lib.all()),
                   key=lambda s: -s.score)
    assert spread(plain, limit=5)[0].name == plain[0].name


# -- pack document 06, mandatory test D ------------------------------------
def test_d_brief_selection_smoke_test(lib):
    """Given a sad + romantic + lonely + warm brief:

    - output must contain 5 ranked ragas
    - each result must include reason + scale
    - result should favour tender/yearning/poignant/warm profiles
    - it should not default to the same raga for every sad brief
    """
    brief = CreativeBrief(situation="love failure", mood="sad romantic",
                          feel="lonely late at night but still warm")
    chosen = rank(brief, lib.all(), limit=5)

    assert len(chosen) == 5
    for item in chosen:
        assert item.reason.strip()
        assert item.raaga.arohanam and item.raaga.avarohanam
        assert 0.0 <= item.score <= 100.0
        assert item.tags

    # Tender / yearning / poignant / warm, in the pack's own vocabulary: the
    # R-G blocks that carry tenderness, or a D-N block that carries pathos.
    tender_blocks = {"R2G2", "R1G2", "R2G3"}
    poignant_or_warm = {"D1N2", "D1N3", "D2N2"}
    favoured = [s for s in chosen
                if s.raaga.rg in tender_blocks or s.raaga.dn in poignant_or_warm]
    assert len(favoured) >= 4, [(s.name, s.raaga.rg, s.raaga.dn) for s in chosen]

    # ... and a different sad brief gets a different answer.
    others = [
        CreativeBrief(mood="sad", feel="devotional, a temple at dusk"),
        CreativeBrief(mood="sad", feel="angry and restless"),
        CreativeBrief(mood="sad", feel="strange, mysterious, searching"),
        CreativeBrief(mood="grief", situation="a funeral", feel="grave, solemn"),
    ]
    tops = {chosen[0].name} | {rank(b, lib.all(), limit=5)[0].name for b in others}
    assert len(tops) >= 3, tops


def test_the_engine_answers_briefs_that_are_not_sad_at_all(lib):
    """A ranking that only works for sadness is a hard-coded raaga in
    disguise; these are the cases that would expose that."""
    for feel, expected in (
            ("bright festive wedding celebration",
             {"Hamsadhwani", "Mohanam", "Shankarabharanam", "Kalyani",
              "Suryakantam", "Madhyamavati"}),
            ("peaceful, a temple at dawn",
             {"Madhyamavati", "Revati", "Mayamalavagowla", "Chakravakam",
              "Shankarabharanam", "Abheri"}),
            ("heroic, urgent, grand",
             {"Chalanata", "Kambhoji", "Sucharitra", "Hatakambari", "Shulini",
              "Nasikabhushani", "Kosalam"})):
        names = {s.name for s in rank(CreativeBrief(feel=feel, mood=""),
                                      lib.all(), limit=5)}
        assert names & expected, (feel, sorted(names))


def test_a_brief_with_nothing_in_it_ranks_nothing(lib):
    """The one case where an empty answer is the honest one; the callers
    turn it into a stated default rather than a silent guess."""
    assert rank(CreativeBrief(mood="", feel="", situation=""), lib.all()) == []
