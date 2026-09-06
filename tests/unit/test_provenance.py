"""Generated material is not evidence.

Training architecture specification sections 2.2 to 2.5, acceptance test E
and non-negotiable rules 1 and 2.
"""
from __future__ import annotations

import sqlite3

import pytest

from raagacomposer.agent.knowledge import KnowledgeRepository, Phrase, Source
from raagacomposer.core import provenance


@pytest.fixture()
def repo(tmp_path):
    r = KnowledgeRepository(tmp_path / "knowledge.db")
    yield r
    r.close()


@pytest.fixture()
def learned(repo):
    source, _ = repo.add_source(Source(locator="rec://1", title="a recording",
                                       raaga="Hamsadhwani",
                                       origin=provenance.HUMAN))
    phrase, _ = repo.add_phrase(Phrase(raaga="Hamsadhwani",
                                       swaras=["S", "R2", "G3"],
                                       source_id=source.id, confidence=0.7))
    return phrase


# --------------------------------------------------------------------------
# the vocabulary
# --------------------------------------------------------------------------
def test_only_material_from_somebody_else_may_be_learned_from():
    assert provenance.may_be_learned_from(provenance.HUMAN)
    assert provenance.may_be_learned_from(provenance.INTERNET)
    assert not provenance.may_be_learned_from(provenance.GENERATED)


def test_an_unrecognised_origin_is_not_trainable():
    """A value nobody checked must not widen what the agent trains on."""
    assert not provenance.may_be_learned_from("")
    assert not provenance.may_be_learned_from("probably fine")


def test_a_phrase_naming_no_source_cannot_be_human():
    assert provenance.coerce(provenance.HUMAN, source_id="") == provenance.GENERATED
    assert provenance.coerce(provenance.HUMAN, source_id="s1") == provenance.HUMAN
    assert provenance.coerce(provenance.INTERNET, source_id="s1") == provenance.INTERNET
    # and a generated phrase stays generated even if it somehow has a source
    assert provenance.coerce(provenance.GENERATED, source_id="s1") == provenance.GENERATED


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
def test_the_store_settles_origin_rather_than_trusting_it(repo):
    made, _ = repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=["P", "N3", "S."],
                                     source_id="", confidence=0.9))
    assert made.origin == provenance.GENERATED
    assert repo.phrase(made.id).origin == provenance.GENERATED


def test_learned_phrases_excludes_what_the_agent_made(repo, learned):
    repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=["P", "N3", "S."],
                           source_id="", confidence=0.9))
    assert len(repo.phrases(raaga="Hamsadhwani")) == 2
    only = repo.learned_phrases(raaga="Hamsadhwani")
    assert [p.id for p in only] == [learned.id]
    assert repo.count_phrases("Hamsadhwani") == 2
    assert repo.count_phrases("Hamsadhwani", learned_only=True) == 1


def test_a_generated_echo_does_not_strengthen_a_learned_phrase(repo, learned):
    """Repeating yourself is not a second witness."""
    repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=list(learned.swaras),
                           source_id="", confidence=0.99))
    after = repo.phrase(learned.id)
    assert after.votes == learned.votes
    assert after.confidence == pytest.approx(learned.confidence)


def test_a_second_real_source_still_strengthens_it(repo, learned):
    """The guard must not break the thing it is guarding."""
    other, _ = repo.add_source(Source(locator="rec://2", title="another",
                                      raaga="Hamsadhwani",
                                      origin=provenance.HUMAN))
    repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=list(learned.swaras),
                           source_id=other.id, confidence=0.8))
    after = repo.phrase(learned.id)
    assert after.votes == learned.votes + 1
    assert after.confidence > learned.confidence


def test_generated_material_is_still_kept_and_composable(repo):
    """Specification 2.3: it is creative material, not contraband."""
    made, is_new = repo.add_phrase(Phrase(raaga="Hamsadhwani",
                                          swaras=["G3", "P", "N3"],
                                          source_id="", confidence=0.6))
    assert is_new
    pool = repo.phrases(raaga="Hamsadhwani")
    assert made.id in [p.id for p in pool]


# --------------------------------------------------------------------------
# migration
# --------------------------------------------------------------------------
def _schema_three_database(path):
    con = sqlite3.connect(str(path))
    con.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO meta VALUES ('schema_version','3');"
        "CREATE TABLE sources (id TEXT PRIMARY KEY, locator TEXT, title TEXT,"
        " performer TEXT, raaga TEXT, content_type TEXT, rights_status TEXT,"
        " provider TEXT, quality REAL, ingested_at REAL, extraction_version TEXT,"
        " confidence REAL, status TEXT, error TEXT, fingerprint TEXT UNIQUE,"
        " notes TEXT);"
        "CREATE TABLE phrases (id TEXT PRIMARY KEY, raaga TEXT, swaras TEXT,"
        " midi TEXT, durations TEXT, function TEXT, source_id TEXT,"
        " confidence REAL, fingerprint TEXT, contour TEXT, tempo REAL,"
        " votes INTEGER, rejected INTEGER, learned_at REAL, notes TEXT);"
        "INSERT INTO phrases VALUES ('p1','Mohanam','[\"S\",\"R2\",\"G3\"]',"
        "'[60,62,64]','[0.4,0.4,0.4]','phrase','s1',0.7,'fp1','',0,1,0,0,'heard');"
        "INSERT INTO phrases VALUES ('p2','Mohanam','[\"P\",\"D2\",\"S.\"]',"
        "'[67,69,72]','[0.4,0.4,0.4]','practice','',0.5,'fp2','',0,1,0,0,'own practice');")
    con.commit()
    con.close()


def test_an_older_database_is_classified_on_open(tmp_path):
    path = tmp_path / "old.db"
    _schema_three_database(path)
    repo = KnowledgeRepository(path)
    try:
        assert repo.schema_version == 4
        by_id = {p.id: p.origin for p in repo.phrases(raaga="Mohanam")}
        assert by_id["p1"] == provenance.HUMAN
        assert by_id["p2"] == provenance.GENERATED
        assert [p.id for p in repo.learned_phrases(raaga="Mohanam")] == ["p1"]
    finally:
        repo.close()


def test_migrating_twice_changes_nothing(tmp_path):
    path = tmp_path / "old.db"
    _schema_three_database(path)
    KnowledgeRepository(path).close()
    repo = KnowledgeRepository(path)
    try:
        assert repo.schema_version == 4
        assert [p.id for p in repo.learned_phrases(raaga="Mohanam")] == ["p1"]
    finally:
        repo.close()


def test_a_recording_can_promote_something_the_agent_only_invented(repo):
    """The rule read the other way round.

    If a real recording turns out to contain a phrase the agent had made
    up, that is a witness and the phrase has now genuinely been heard.
    Leaving it marked generated would throw the evidence away.
    """
    made, _ = repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=["S", "G3", "P"],
                                     source_id="", confidence=0.5))
    assert made.origin == provenance.GENERATED
    assert repo.learned_phrases(raaga="Hamsadhwani") == []

    source, _ = repo.add_source(Source(locator="rec://3", title="a recording",
                                       raaga="Hamsadhwani",
                                       origin=provenance.HUMAN))
    repo.add_phrase(Phrase(raaga="Hamsadhwani", swaras=["S", "G3", "P"],
                           source_id=source.id, confidence=0.8))
    after = repo.phrase(made.id)
    assert after.origin == provenance.HUMAN
    assert after.source_id == source.id
    assert [p.id for p in repo.learned_phrases(raaga="Hamsadhwani")] == [made.id]
