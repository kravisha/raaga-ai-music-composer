"""The Producer makes a whole rough song, and the Critic reviews every stage.

Real controller, real synthesis, a fake Codex transport.  These prove the
mechanical path and the bookkeeping - which stages Codex accepted, which the
Producer decided - not that anything sounds good.
"""
import json
import time
from pathlib import Path

import pytest

from raagacomposer.core.models import SectionKind
from raagacomposer.production.contracts import (PRODUCTION_STAGES,
                                                CodexCliTransport,
                                                CodexCritic, ProjectState)
from raagacomposer.production.producer import Producer, locate_codex


class FakeCodex:
    """An explicitly fake transport: accepts unless the policy says otherwise."""

    provider = "codex"
    model = "test-double"

    def __init__(self, policy=None):
        self.policy = policy or (lambda stage, packet, calls: None)
        self.calls = []

    def __call__(self, prompt, schema):
        packet = json.loads(prompt.split("Evidence packet:\n", 1)[1])
        self.calls.append(packet["stage"])
        response = dict(request_id=packet["request_id"], stage=packet["stage"],
                        accept=True, blocked=False,
                        reason="The supplied evidence supports this stage",
                        revisions=[], strengths=["The returning phrase"],
                        lessons=[], uncertainty=["Nothing has been heard"],
                        evidence=[])
        change = self.policy(packet["stage"], packet, self.calls)
        if change:
            response.update(change)
        return response


def drive(app, producer, timeout: float = 600.0) -> None:
    """Pump the way the window's timer does, until the production ends."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.pump()
        if producer.finished:
            app.pump()
            return
        time.sleep(0.02)
    active = [j.description for j in app.jobs.active_jobs()]
    raise TimeoutError(f"production stuck in {producer.phase} at the "
                       f"{producer.stage}: {active}")


def a_whole_song_brief(app, title: str) -> None:
    app.new_project(title, write=False)
    app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                     tala="", situation="A hopeful reunion after a long separation",
                     notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                           "Charanam and Ending.")
    app.select_raaga("Hamsadhwani")


SUNG = {SectionKind.PALLAVI, SectionKind.ANUPALLAVI, SectionKind.CHARANAM}


def test_the_producer_makes_a_whole_song_with_every_stage_reviewed(app):
    a_whole_song_brief(app, "Whole song by the Producer")
    codex = FakeCodex()
    app.critic = CodexCritic(codex)
    producer = app.produce_song(seed=104)
    drive(app, producer)
    assert producer.phase == "done", producer.report()

    # Every stage was reviewed, in order, once, and Codex accepted them all.
    assert codex.calls == list(PRODUCTION_STAGES)
    journal = ProjectState.load(producer.journal_path)
    for stage in PRODUCTION_STAGES:
        assert journal.is_accepted(stage, producer.refs[stage]), (stage, producer.refs)
    assert producer.accepted == 7 and producer.decisions == 0

    # And a whole song exists: tune with the asked-for sections, words on the
    # sung ones, a studio vocal, a beat in the arrangement, a full mix on disk.
    melody = app.project.melody()
    kinds = {s.kind for s in melody.sections}
    assert {SectionKind.PRELUDE, SectionKind.INTERLUDE, SectionKind.OUTRO} | SUNG <= kinds
    lyrics = app.project.lyrics_version()
    sung_ids = {s.id for s in melody.sections if s.kind in SUNG}
    assert sung_ids <= {line.section_id for line in lyrics.lines}
    take = app.project.vocal_master
    assert take and Path(take.audio_path).is_file()
    assert app.project.beat() and app.project.arrangement().tracks
    mix = app.project.latest_mix("full")
    assert mix and Path(mix.audio_path).is_file()
    assert mix.duration >= melody.duration - 0.05
    report = producer.report()
    assert "Codex accepted 7 of 7 stages" in report
    assert Path(producer.journal_path.with_name("production_report.txt")).is_file()
    # The production is part of the song's record, and the song was saved.
    assert any(h.action == "production.done" for h in app.project.history)
    assert not app.dirty


def test_a_revision_is_asked_of_the_tune_and_the_advice_reaches_the_agent(app):
    a_whole_song_brief(app, "Revised once")
    advice = "Leave the singer room at the cadence of the Pallavi"

    def policy(stage, packet, calls):
        if stage == "tune" and calls.count("tune") == 1:
            return {"accept": False, "revisions": [advice]}
        return None

    codex = FakeCodex(policy)
    app.critic = CodexCritic(codex)
    producer = app.produce_song(seed=104, max_rounds=2)
    drive(app, producer)
    assert producer.phase == "done", producer.report()

    # Two tunes were written, the second after the advice; Codex accepted it.
    assert len(app.project.melodies) == 2
    assert codex.calls.count("tune") == 2
    journal = ProjectState.load(producer.journal_path)
    verdicts = [(r.round, r.verdict) for r in journal.records if r.stage == "tune"]
    assert verdicts == [(1, "revise"), (2, "accept")], verdicts
    assert journal.is_accepted("tune", "tune:v2")
    assert not journal.is_accepted("tune", "tune:v1")
    # The advice went to the agent through the feedback door, not into a log
    # nobody reads.
    fed = [h for h in app.project.history if h.action == "agent.feedback"]
    assert fed and advice in fed[-1].description, [h.description for h in fed]


def test_without_codex_the_song_is_still_made_and_says_so(app, tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: None)
    a_whole_song_brief(app, "Unreviewed")
    called = []
    app.critic = CodexCritic(CodexCliTransport(
        tmp_path / "critic", run=lambda *a, **k: called.append(a)))
    assert not app.critic.available
    producer = app.produce_song(seed=104)
    drive(app, producer)
    assert producer.phase == "done", producer.report()
    assert called == [], "an unavailable reviewer must not be spawned"

    journal = ProjectState.load(producer.journal_path)
    for stage in PRODUCTION_STAGES:
        assert not journal.is_accepted(stage, producer.refs[stage])
        decided = [r for r in journal.records
                   if r.stage == stage and r.role == "producer"]
        assert decided and "unavailable" in decided[-1].rationale, stage
    assert producer.accepted == 0 and producer.decisions == 7
    assert all(r.role == "producer" for r in journal.records), \
        "an unavailable reviewer must not be asked, so no critic record can exist"
    report = producer.report()
    assert "Codex accepted 0 of 7 stages" in report and "unavailable" in report
    assert Path(app.project.latest_mix("full").audio_path).is_file()


def test_a_production_told_to_stop_on_a_blocked_review_stops(app, tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: None)
    a_whole_song_brief(app, "Stops when unreviewed")
    app.critic = CodexCritic(CodexCliTransport(tmp_path / "critic"))
    app.settings.production_on_blocked = "stop"
    producer = app.produce_song(seed=104)
    drive(app, producer, timeout=120)
    assert producer.phase == "failed" and producer.stage == "brief"
    assert "unavailable" in producer.error, producer.error
    assert not app.project.melodies, "a stopped production must not compose"
    journal = ProjectState.load(producer.journal_path)
    assert not journal.records, "nothing was reviewed and nothing was decided"


def test_an_exhausted_budget_is_a_producer_decision_not_an_acceptance(app):
    a_whole_song_brief(app, "Never good enough")
    codex = FakeCodex(lambda stage, packet, calls: (
        {"accept": False, "revisions": ["More variation in the Charanam"]}
        if stage == "tune" else None))
    app.critic = CodexCritic(codex)
    producer = app.produce_song(seed=104, max_rounds=1)
    drive(app, producer)
    assert producer.phase == "done", producer.report()
    assert codex.calls.count("tune") == 1, "a spent budget must not call the model"
    assert len(app.project.melodies) == 1
    journal = ProjectState.load(producer.journal_path)
    tune = [(r.role, r.verdict) for r in journal.records if r.stage == "tune"]
    assert tune == [("critic", "revise"), ("critic", "blocked"),
                    ("producer", "accept")], tune
    assert not journal.is_accepted("tune", "tune:v1")
    assert producer.accepted == 6 and producer.decisions == 1
    assert "spent" in producer.report()


def test_switching_songs_stops_the_production(app):
    a_whole_song_brief(app, "Abandoned")
    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    assert not producer.finished and producer.stage == "brief"
    # Let it get as far as writing the tune, then leave the song.
    for _ in range(50):
        app.pump()
        if producer.stage == "tune":
            break
        time.sleep(0.02)
    assert producer.stage == "tune", producer.report()
    app.new_project("The next song", write=False)
    drive(app, producer, timeout=120)
    assert producer.phase == "failed"
    assert "changed" in producer.error, producer.error
    assert not app.project.melodies, "the abandoned tune landed in the new song"


def test_producing_twice_at_once_is_refused(app):
    a_whole_song_brief(app, "One at a time")
    app.critic = CodexCritic(FakeCodex())
    first = app.produce_song(seed=104)
    assert not first.finished
    assert app.produce_song(seed=105) is first
    first.cancel("test over")
    drive(app, first, timeout=120)
    assert first.phase == "cancelled"


def test_locate_codex_prefers_the_configured_path(tmp_path, monkeypatch):
    monkeypatch.setattr("raagacomposer.production.producer.shutil.which",
                        lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert locate_codex("C:/somewhere/codex.exe") == "C:/somewhere/codex.exe"
    assert locate_codex("") == ""
    old = tmp_path / "OpenAI" / "Codex" / "bin" / "aaaa" / "codex.exe"
    new = tmp_path / "OpenAI" / "Codex" / "bin" / "bbbb" / "codex.exe"
    for path, when in ((old, 1_000_000), (new, 2_000_000)):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        import os
        os.utime(path, (when, when))
    assert locate_codex("") == str(new)
