"""What may be reached, and what must be said when it may not.

Specification section 4 and section 20 rules 9 and 10.  These are the tests
that keep the feature honest: they assert that a source on the network is not
fetched, that being unable to reach something is reported rather than papered
over, and that the two honest ways forward are offered.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.training.access import SourceAccessService
from raagacomposer.training.models import Accessibility, LearningSource
from tests.conftest import ANALYSIS_SR, sung_signal

pytestmark = pytest.mark.unit


@pytest.fixture
def access(training_store) -> SourceAccessService:
    return SourceAccessService(training_store)


# --------------------------------------------------------------------------
# what we do not do
# --------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=DoTQ3ZYK-Qw",
    "https://youtu.be/DoTQ3ZYK-Qw",
    "https://example.org/lessons/kambhoji",
    "http://some.site/a-lesson.html",
])
def test_a_source_on_the_network_is_not_fetched(access, url):
    """Section 20 rule 10.  Nothing here downloads, and the decision says so
    rather than leaving the caller to assume."""
    decision = access.check(LearningSource(title="A lesson", url=url))
    assert decision.status == Accessibility.METADATA_ONLY
    assert not decision.analysable
    assert "has not been analysed" in decision.reason


def test_an_unreachable_source_offers_the_two_honest_ways_forward(access):
    decision = access.check(LearningSource(
        title="A lesson", url="https://example.org/x"))
    assert "Upload the source file manually" in decision.offers
    assert "Provide transcript" in decision.offers


def test_a_source_with_nowhere_to_look_is_not_accessible(access):
    decision = access.check(LearningSource(title="A lesson"))
    assert decision.status == Accessibility.NOT_ACCESSIBLE
    assert not decision.analysable


def test_an_unknown_scheme_is_unsupported_rather_than_attempted(access):
    decision = access.check(LearningSource(
        title="x", url="ftp://example.org/thing.wav"))
    assert decision.status == Accessibility.UNSUPPORTED
    assert not decision.analysable


# --------------------------------------------------------------------------
# what we do
# --------------------------------------------------------------------------
def test_the_applications_own_exercises_are_accessible(access):
    decision = access.check(LearningSource(
        title="Kambhoji: identity and scale",
        url="raaga-exercise://Kambhoji/identity"))
    assert decision.status == Accessibility.ACCESSIBLE
    assert decision.representation == "exercise"


def test_a_file_the_creator_supplied_is_read(access, tmp_path):
    path = tmp_path / "Keeravani-alapana.wav"
    sf.write(path, sung_signal(2.0), ANALYSIS_SR)
    decision = access.check(LearningSource(title="mine",
                                           local_path=str(path)))
    assert decision.status == Accessibility.ACCESSIBLE
    assert decision.representation == "audio"


def test_a_supplied_file_beats_the_url_it_came_from(access, tmp_path):
    """The creator handing over the file is exactly how an unreachable source
    becomes learnable, so the local copy has to win."""
    path = tmp_path / "Keeravani.wav"
    sf.write(path, sung_signal(2.0), ANALYSIS_SR)
    decision = access.check(LearningSource(
        title="a lesson", url="https://youtu.be/DoTQ3ZYK-Qw",
        local_path=str(path)))
    assert decision.status == Accessibility.ACCESSIBLE
    assert decision.representation == "audio"


def test_a_supplied_transcript_makes_a_network_source_readable(access):
    decision = access.check(LearningSource(
        title="a lesson", url="https://example.org/v",
        metadata={"transcript": "Kambhoji is a janya of Harikambhoji."}))
    assert decision.status == Accessibility.TRANSCRIPT
    assert decision.representation == "transcript"


def test_a_transcript_file_on_disk_is_read(access, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Sa ri ga ma pa dha ni sa", encoding="utf-8")
    decision = access.check(LearningSource(title="notes",
                                           local_path=str(path)))
    assert decision.representation == "transcript"


# --------------------------------------------------------------------------
# the limitation we are honest about
# --------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ".webm", ".mov"])
def test_a_video_file_asks_for_its_audio_rather_than_failing_later(access,
                                                                   tmp_path,
                                                                   suffix):
    """There is no demuxer in this application.  Saying so up front is better
    than a decode error three phases later that reads like a bug."""
    path = tmp_path / f"lesson{suffix}"
    path.write_bytes(b"not really a video")
    decision = access.check(LearningSource(title="x", local_path=str(path)))
    assert decision.status == Accessibility.USER_FILE_REQUIRED
    assert "extract the audio" in decision.reason
    assert not decision.analysable


def test_a_file_that_is_not_there_is_reported(access, tmp_path):
    decision = access.check(LearningSource(
        title="x", url=(tmp_path / "missing.wav").as_uri()))
    assert not decision.analysable


def test_checking_access_records_what_was_decided(access, training_store):
    """Section 16: the decision is part of the provenance, not a passing
    thought."""
    source = training_store.save_candidate(LearningSource(
        title="x", url="https://example.org/v"))
    decision = access.check(source)
    access.record_provenance(source, "run_1", decision)
    trail = training_store.audit_trail(run_id="run_1")
    assert any(row["kind"] == "access.checked" for row in trail)
    assert training_store.candidate(source.source_id).accessibility_status \
        == Accessibility.METADATA_ONLY
