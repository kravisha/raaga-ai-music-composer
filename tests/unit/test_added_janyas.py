"""The janya raagas added for a real collection, and what they may claim.

Every one of these appears by name in recordings the creator has, and none
was in the library, so those files could not be filed at all.

They arrive as scales and nothing else.  The arohanam and avarohanam are a
matter of record; prayogas, resting notes and ornament are matters of
practice, and asserting them would be putting my word where a musician's
belongs.  ``Raaga.scale_only`` is derived from their absence, so the
application already says so and suggests these at lower confidence - and
the listening path can fill them in from real recordings as evidence.
"""
from __future__ import annotations

import pytest

from raagacomposer.raaga.library import SWARA_SEMITONES, RaagaLibrary

pytestmark = pytest.mark.unit

ADDED = ["Suddha Saveri", "Suddha Dhanyasi", "Nalinakanti", "Vasantha",
         "Behag", "Anandabhairavi", "Kaapi", "Sahana", "Reetigowla",
         "Dwijavanti", "Darbari Kanada"]

#: The spellings that actually turn up in downloaded filenames.  A raaga the
#: library cannot recognise under the creator's spelling is a raaga it does
#: not have, as far as the manifest is concerned.
SPELLINGS = {
    "Reetigowlai": "Reetigowla",
    "Anandhabhairavi": "Anandabhairavi",
    "Dwijavanthi": "Dwijavanti",
    "Nalinakanthi": "Nalinakanti",
    "Shuddha Dhanyasi": "Suddha Dhanyasi",
    "Shuddha Saveri": "Suddha Saveri",
    "Kapi": "Kaapi",
    "Bihag": "Behag",
}


@pytest.fixture(scope="module")
def lib():
    return RaagaLibrary()


@pytest.mark.parametrize("name", ADDED)
def test_the_raaga_is_there(lib, name):
    assert lib.get(name) is not None, f"{name} is missing from the library"


@pytest.mark.parametrize("spelling,canonical", sorted(SPELLINGS.items()))
def test_the_creators_spelling_finds_it(lib, spelling, canonical):
    found = lib.get(spelling)
    assert found is not None, f"{spelling!r} does not resolve"
    assert found.name == canonical


@pytest.mark.parametrize("name", ADDED)
def test_it_claims_only_its_scale(lib, name):
    """No invented prayogas, jeeva, nyasa or gamaka.

    If one of these ever gains them it should be because a musician curated
    them or the agent heard them - not because this file grew.
    """
    raaga = lib.get(name)
    assert raaga.scale_only, f"{name} claims idiom it was not given"
    assert not raaga.prayogas and not raaga.jeeva
    assert not raaga.nyasa and not raaga.gamaka


@pytest.mark.parametrize("name", ADDED)
def test_the_scale_is_structurally_sound(lib, name):
    raaga = lib.get(name)
    assert raaga.arohanam and raaga.avarohanam, name
    for swara in raaga.arohanam + raaga.avarohanam:
        assert swara.rstrip("+-") in SWARA_SEMITONES, f"{name}: {swara!r}"
    assert raaga.arohanam[0].rstrip("+-") == "S", f"{name} does not start on Sa"
    assert raaga.avarohanam[-1].rstrip("+-") == "S", f"{name} does not end on Sa"
    assert raaga.arohanam[-1] == "S+", f"{name}'s ascent stops short of the octave"
    assert raaga.avarohanam[0] == "S+", f"{name}'s descent does not start above"
    assert len(raaga.allowed) >= 5, f"{name} has too few swaras to be a raaga"


@pytest.mark.parametrize("name", ADDED)
def test_a_tune_can_actually_be_generated_in_it(lib, name):
    """A scale nobody can compose in is not much of an addition."""
    from raagacomposer.music import melody

    raaga = lib.get(name)
    tune = melody.generate(raaga, melody.MelodyOptions(
        seed=7, duration_target=20.0, tonic_midi=60))
    assert tune.notes, f"nothing could be generated in {name}"
    allowed = set(raaga.allowed)
    from raagacomposer.raaga.library import parse_swara
    outside = [n.swara for n in tune.notes
               if parse_swara(n.swara)[0] not in allowed]
    assert not outside, f"{name}: generated notes outside the raaga: {outside[:5]}"
