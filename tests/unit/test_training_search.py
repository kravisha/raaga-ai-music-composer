"""Finding material - specification section 5.

The interesting behaviour here is not that a search returns rows; it is what
it does with a name spelled a different way, a source it may not open, and the
same lesson arriving twice.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.training.models import Accessibility, SearchQuery
from raagacomposer.training.search import (LearningSourceSearchService,
                                           LocalLibraryProvider,
                                           ReferenceExerciseProvider,
                                           WebLeadProvider, match_raaga,
                                           tokenize)
from tests.conftest import ANALYSIS_SR, sung_signal

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# reading the phrase
# --------------------------------------------------------------------------
@pytest.mark.parametrize("phrase,expected", [
    ("Kamboji raga beginner lesson", "Kambhoji"),      # the spec's own spelling
    ("Kambhoji alapana", "Kambhoji"),
    ("learn Keeravani arohanam", "Keeravani"),
    ("Mayamalavagoula varisai", "Mayamalavagowla"),    # library alias spelling
    ("Yaman lessons", "Kalyani"),                      # Hindustani alias
    ("learn Bhoop", "Mohanam"),
    ("Kirwani gamaka", "Keeravani"),
])
def test_a_raaga_is_recognised_however_it_is_spelled(phrase, expected, raagas):
    """Roman transliterations of one raaga differ exactly where aspirates and
    doubled vowels do, and the creator should not have to guess ours."""
    found = match_raaga(phrase, raagas)
    assert found is not None and found.name == expected


def test_a_phrase_about_no_raaga_matches_none(raagas):
    assert match_raaga("Adi tala lessons for beginners", raagas) is None
    assert match_raaga("", raagas) is None


def test_tokenize_drops_the_words_every_search_contains():
    assert "lesson" not in tokenize("Kambhoji beginner lesson")
    assert "kambhoji" in tokenize("Kambhoji beginner lesson")


# --------------------------------------------------------------------------
# the service
# --------------------------------------------------------------------------
def test_the_acceptance_phrase_returns_about_ten_results(training_store,
                                                         raagas):
    service = LearningSourceSearchService(training_store, raagas)
    results = service.search(SearchQuery(phrase="Kamboji raga beginner lesson"))
    assert 8 <= len(results) <= 10
    assert all("Kambhoji" in r.title for r in results)


def test_results_are_saved_so_they_survive_a_restart(training_store, raagas):
    service = LearningSourceSearchService(training_store, raagas)
    results = service.search(SearchQuery(phrase="Keeravani phrases"))
    stored = training_store.candidate(results[0].source_id)
    assert stored is not None and stored.title == results[0].title


def test_a_source_we_can_analyse_outranks_one_we_can_only_name(training_store,
                                                               raagas):
    """Section 5.4 asks for ranking by usefulness *for learning*.  A perfectly
    titled source we may not open teaches nobody anything."""
    def finder(phrase, limit):
        return [{"title": "Keeravani characteristic phrases prayoga lesson",
                 "url": "https://example.org/perfect-title",
                 "author": "somebody"}]

    service = LearningSourceSearchService(
        training_store, raagas,
        providers=[ReferenceExerciseProvider(raagas),
                   WebLeadProvider(True, finder)])
    # Room for everything, so this measures the ranking rather than the trim.
    results = service.search(SearchQuery(
        phrase="Keeravani characteristic phrases prayoga", max_results=40))
    lead = next(r for r in results if r.provider == "web")
    best = results[0]
    assert best.provider == "exercises"
    assert best.relevance_score > lead.relevance_score
    # And with only ten slots, the lead is the one that loses its place.
    top_ten = service.search(SearchQuery(
        phrase="Keeravani characteristic phrases prayoga", max_results=10))
    assert all(r.provider == "exercises" for r in top_ten)


def test_the_web_provider_is_off_unless_it_is_turned_on(training_store,
                                                        raagas):
    off = WebLeadProvider(False, lambda p, n: [{"title": "x", "url": "y"}])
    assert not off.available()
    assert off.search(SearchQuery(phrase="x"), None, 5) == []


def test_a_web_result_is_a_lead_and_says_so(training_store, raagas):
    """Section 4: nothing is fetched, and the row must not imply otherwise."""
    provider = WebLeadProvider(
        True, lambda p, n: [{"title": "A lesson",
                             "url": "https://example.org/v",
                             "description": "Teaches Kambhoji."}])
    lead = provider.search(SearchQuery(phrase="Kambhoji"), None, 5)[0]
    assert lead.accessibility_status == Accessibility.METADATA_ONLY
    assert not lead.can_be_analysed
    assert "nothing has been fetched" in lead.description.lower()


def test_a_provider_that_throws_does_not_lose_the_whole_search(training_store,
                                                               raagas):
    class Broken(ReferenceExerciseProvider):
        name = "broken"

        def search(self, query, raaga, limit):
            raise RuntimeError("provider exploded")

    service = LearningSourceSearchService(
        training_store, raagas,
        providers=[Broken(raagas), ReferenceExerciseProvider(raagas)])
    assert service.search(SearchQuery(phrase="Kambhoji lesson"))


def test_the_same_lesson_twice_is_listed_once(training_store, raagas):
    class Twice(ReferenceExerciseProvider):
        name = "twice"

        def search(self, query, raaga, limit):
            found = super().search(query, raaga, limit)
            return found + found

    service = LearningSourceSearchService(training_store, raagas,
                                          providers=[Twice(raagas)])
    results = service.search(SearchQuery(phrase="Kambhoji lesson"))
    assert len({r.url for r in results}) == len(results)


def test_excluded_words_remove_a_result(training_store, raagas):
    service = LearningSourceSearchService(training_store, raagas)
    everything = service.search(SearchQuery(phrase="Kambhoji"))
    assert any("gamaka" in r.title.lower() for r in everything)
    filtered = service.search(
        SearchQuery(phrase="Kambhoji", exclude_keywords=["gamaka"]))
    assert not any("gamaka" in r.title.lower() for r in filtered)


def test_the_difficulty_filter_narrows_the_topics(training_store, raagas):
    service = LearningSourceSearchService(training_store, raagas)
    results = service.search(SearchQuery(phrase="Kambhoji",
                                         difficulty="beginner"))
    assert results
    assert all(r.metadata.get("difficulty") == "beginner" for r in results)


def test_a_source_filter_uses_only_that_provider(training_store, raagas):
    service = LearningSourceSearchService(training_store, raagas)
    results = service.search(SearchQuery(phrase="Kambhoji",
                                         source_filter="exercises"))
    assert results and {r.provider for r in results} == {"exercises"}


# --------------------------------------------------------------------------
# the creator's own folder
# --------------------------------------------------------------------------
def test_the_learning_folder_is_searched_and_is_analysable(tmp_path, raagas):
    folder = tmp_path / "learning" / "Keeravani"
    folder.mkdir(parents=True)
    sf.write(folder / "Keeravani-alapana.wav", sung_signal(3.0), ANALYSIS_SR)

    provider = LocalLibraryProvider(tmp_path / "learning", raagas)
    assert provider.available()
    found = provider.search(SearchQuery(phrase="Keeravani alapana"),
                            raagas.require("Keeravani"), 5)
    assert len(found) == 1
    assert found[0].can_be_analysed
    assert found[0].accessibility_status == Accessibility.ACCESSIBLE
    assert found[0].local_path


def test_a_folder_that_is_not_there_is_simply_unavailable(tmp_path, raagas):
    provider = LocalLibraryProvider(tmp_path / "nothing-here", raagas)
    assert not provider.available()
    assert provider.search(SearchQuery(phrase="x"), None, 5) == []
