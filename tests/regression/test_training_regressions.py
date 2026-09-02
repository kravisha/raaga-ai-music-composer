"""Defects found while building the Training tab, each named.

A failure here should explain itself without archaeology, so every test says
what went wrong and why it mattered rather than only asserting the fix.
"""
from __future__ import annotations

import pytest
import soundfile as sf

from raagacomposer.training.knowledge_base import KnowledgeBaseService
from raagacomposer.training.models import (Accessibility, KnowledgeEntry,
                                           LearningRun, LearningSource,
                                           RunStatus)
from raagacomposer.training.pipeline import METADATA_ONLY_NOTICE
from raagacomposer.training.search import (LocalLibraryProvider, match_raaga)
from raagacomposer.training.models import SearchQuery
from tests.conftest import ANALYSIS_SR, sung_signal

pytestmark = pytest.mark.regression


def test_reg_the_specifications_own_spelling_found_nothing(training, raagas):
    """The acceptance test asks for "Kamboji"; the library calls it
    "Kambhoji".  Matched literally, the demonstration in section 19 returned
    zero results - the one search the specification actually names.

    Roman transliterations of a raaga differ exactly at the aspirates and the
    doubled vowels, so the creator must not have to guess our spelling.
    """
    assert match_raaga("Kamboji raga beginner lesson", raagas).name \
        == "Kambhoji"
    results = training.search("Kamboji raga beginner lesson")
    assert len(results) >= 8


def test_reg_a_file_in_a_flat_folder_belonged_to_no_raaga(tmp_path, raagas):
    """The learning-folder provider had a stub where the raaga should have
    been worked out from the filename, so a file that was not found by a
    raaga-specific search arrived with no raaga attached and could never be
    turned into phrases for anything.
    """
    folder = tmp_path / "learning"
    folder.mkdir()
    sf.write(folder / "Keeravani-alapana.wav", sung_signal(2.0), ANALYSIS_SR)

    provider = LocalLibraryProvider(folder, raagas)
    # A search that names no raaga at all: the file itself has to say.
    found = provider.search(SearchQuery(phrase="alapana"), None, 5)
    assert found
    assert found[0].metadata.get("raaga") == "Keeravani"


def test_reg_knowledge_without_provenance_must_not_be_stored(training_store,
                                                             raagas):
    """Section 16 requires every item to say where it came from.  An entry
    with no source or run would satisfy the schema and be unanswerable
    afterwards, so it is refused at the door rather than stored and puzzled
    over later.
    """
    service = KnowledgeBaseService(training_store, raagas)
    source = LearningSource(source_id="src_1", title="x")
    stored = service.store_all([
        KnowledgeEntry(subject="a", normalized_statement="orphan",
                       category="scale"),                    # no source, no run
        KnowledgeEntry(subject="b", normalized_statement="proper",
                       category="scale", source_id="src_1", run_id="run_1"),
    ], source, "run_1")
    assert len(stored) == 1
    assert training_store.knowledge_count() == 1
    assert training_store.search_knowledge()[0].normalized_statement == "proper"


def test_reg_the_metadata_only_notice_is_the_specifications_wording(training):
    """The specification names this string exactly.  Paraphrasing it makes a
    report that a reader skims look like a report of findings.
    """
    source = training.store.save_candidate(LearningSource(
        title="A lesson", url="https://youtu.be/DoTQ3ZYK-Qw",
        accessibility_status=Accessibility.METADATA_ONLY, provider="web"))
    training.queue.enqueue([source], "x")
    report = training.learn_one_now()

    assert METADATA_ONLY_NOTICE == "METADATA ONLY - CONTENT NOT ANALYZED"
    assert METADATA_ONLY_NOTICE in report.summary
    assert METADATA_ONLY_NOTICE in training.render_report(report.run_id)


def test_reg_a_relearn_must_not_destroy_the_earlier_report(training):
    """Section 10 is explicit that relearning creates a new run rather than
    overwriting the previous one.  Keying reports by source rather than by run
    would have silently replaced the history the creator was told they had.
    """
    results = training.search("Keeravani characteristic phrases")
    first = training.add_to_queue([results[0].source_id])[0]
    training.learn_one_now()
    before = training.render_report(first.run_id)

    second = training.relearn(results[0].source_id)
    training.learn_one_now()

    assert training.render_report(first.run_id) == before
    assert second.run_id != first.run_id
    assert training.store.report(second.run_id) is not None


def test_reg_an_exercise_must_not_be_put_through_the_speech_gate(training):
    """The Training tab reuses the recording preparation built for real
    lessons.  Applied to material the application rendered itself - one clean
    voice, no drone, nobody talking - it could only remove good phrases.
    """
    results = training.search("Keeravani characteristic phrases")
    training.add_to_queue([results[0].source_id])
    report = training.learn_one_now()
    assert report.learned
    assert not any("silenced as speech" in limit
                   for limit in report.honest_limits)


def test_reg_a_source_that_fails_still_leaves_a_report(training):
    """Rule 4 has no exception for failure.  Without this the queue could show
    'Failed' with nothing anywhere saying what had gone wrong.
    """
    broken = training.store.save_candidate(LearningSource(
        title="Broken", url="raaga-exercise://NotARaaga/identity",
        metadata={"raaga": "NotARaaga", "topic": "identity"}))
    training.queue.enqueue([broken], "x")
    training.learn_one_now()
    run = training.store.runs()[0]
    assert run.status == RunStatus.FAILED
    assert training.store.report(run.run_id) is not None
