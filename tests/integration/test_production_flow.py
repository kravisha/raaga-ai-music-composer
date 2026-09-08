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
                                                CodexCritic, ProjectState,
                                                StageRecord, Verdict)
from raagacomposer.production.producer import Producer, locate_codex


class FakeCodex:
    """An explicitly fake transport: accepts unless the policy says otherwise."""

    provider = "codex"
    model = "test-double"

    def __init__(self, policy=None):
        self.policy = policy or (lambda stage, packet, calls: None)
        self.calls = []

    def __call__(self, prompt, schema, *, cancelled=None):
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
    # The advice reached the agent as a lesson with Codex's name on it - not
    # as the creator's testimony, and not into a log nobody reads.
    noted = [h for h in app.project.history if h.action == "critic.advice"]
    assert noted and advice in noted[-1].description, \
        [h.description for h in app.project.history]
    lessons = [l for l in app.agent.repo.lessons(raaga="Hamsadhwani")
               if l.method == "codex critic"]
    assert lessons and all(l.dimension == "critic" and l.confidence < 0.9
                           for l in lessons), \
        [(l.kind, l.method, l.dimension, l.confidence) for l in lessons]
    assert not any(l.method == "creator feedback" or l.dimension == "creator"
                   for l in app.agent.repo.lessons(raaga="Hamsadhwani")), \
        "Codex's advice was filed as the creator's own words"


def test_without_codex_the_song_is_still_made_and_says_so(app, tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: None)
    a_whole_song_brief(app, "Unreviewed")
    called = []
    app.critic = CodexCritic(CodexCliTransport(
        tmp_path / "critic", popen=lambda *a, **k: called.append(a)))
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
    # Built directly rather than through the setting: saving a project
    # persists the whole settings object into the shared test home, and a
    # "stop" written there would reach every later test's Settings.load().
    producer = Producer(app, app.critic, seed=104, on_blocked="stop")
    app.producer = producer
    assert producer.start()
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


# ----------------------------------------------------------------------
# Boundaries Arya found by independent review of the first integration
# (their producer_boundary_cases.py, ported here so they run in the suite).
# ----------------------------------------------------------------------
import threading


class GateCodex(FakeCodex):
    """Accepts, but only once the test lets it: the review is held open so
    the song can change underneath it."""

    def __init__(self, policy=None):
        super().__init__(policy)
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, prompt, schema, *, cancelled=None):
        self.entered.set()
        assert self.release.wait(8), "the test must release the reviewer"
        return super().__call__(prompt, schema, cancelled=cancelled)


def pump_until(app, predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.pump()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("the production did not reach the expected phase")


def test_a_second_production_continues_the_journal_it_finds(app):
    """A restart used to build a fresh ProjectState beside an existing
    production.json; every later save then hit the stale-journal guard and
    _save_journal swallowed it, so the new history never reached disk."""
    a_whole_song_brief(app, "Produced twice")
    first = Producer(app, CodexCritic(FakeCodex()))
    assert first.start()
    first.state.add_record(StageRecord(
        stage="brief", role="critic", artifact_ref="brief:first",
        rationale="First review preserved", verdict="accept", round=1,
        provider="codex"))
    first.cancel("first run stopped")
    assert len(ProjectState.load(first.journal_path).records) == 1

    second = Producer(app, CodexCritic(FakeCodex()))
    assert second.start(), second.error
    assert second.state.revision == 1, "a restart is a new revision of the same journal"
    second.state.add_record(StageRecord(
        stage="brief", role="critic", artifact_ref="brief:second",
        rationale="Second review must survive", verdict="accept", round=1,
        revision=second.state.revision, provider="codex"))
    second.cancel("second run stopped")
    saved = ProjectState.load(second.journal_path)
    assert [r.rationale for r in saved.records] == \
        ["First review preserved", "Second review must survive"], second.events
    assert not saved.is_accepted("brief", "brief:first"), \
        "an earlier revision's acceptance must not count for this one"


def test_a_brief_edited_during_its_review_is_not_accepted(app):
    """The reviewer answered about a snapshot; the creator changed the brief
    while it thought.  The verdict is about a brief that no longer exists."""
    a_whole_song_brief(app, "Edited under review")
    gate = GateCodex()
    app.critic = CodexCritic(gate)
    producer = app.produce_song(seed=104)
    try:
        pump_until(app, gate.entered.is_set)
        assert producer.phase == "reviewing" and producer.stage == "brief"
        app.update_brief(situation="A farewell with no hope of reunion")
        gate.release.set()
        pump_until(app, lambda: producer.phase != "reviewing")
        assert producer.accepted == 0, "the old brief was accepted after the live brief changed"
        assert not any(r.verdict == "accept" and r.provider == "codex"
                       for r in producer.state.records)
        assert producer.phase == "failed" and "changed" in producer.error, producer.error
        assert not app.project.melodies, "production must not have gone on to the tune"
    finally:
        gate.release.set()
        if not producer.finished:
            producer.cancel("test over")


def test_codex_advice_is_never_filed_as_the_creators_words(app):
    a_whole_song_brief(app, "Advice provenance")
    producer = Producer(app, CodexCritic(FakeCodex()))
    assert producer.start()
    producer.stage = "tune"
    before = {l.id for l in app.agent.repo.lessons(raaga="Hamsadhwani")}
    producer._apply_advice(Verdict(
        False, "Variation is required", provider="codex",
        revisions=["The melody is too repetitive and mechanical; vary the answer phrase"]))
    added = [l for l in app.agent.repo.lessons(raaga="Hamsadhwani") if l.id not in before]
    assert added, "the advice must reach the agent somewhere"
    wrong = [l for l in added if l.method == "creator feedback" or l.dimension == "creator"
             or l.confidence >= 0.9]
    assert not wrong, [(l.kind, l.method, l.dimension, l.confidence) for l in wrong]
    assert all(l.method == "codex critic" for l in added)
    producer.cancel("test over")


def test_stop_policy_holds_when_the_budget_is_spent(app):
    """Exhaustion took the proceed path directly, bypassing on_blocked."""
    a_whole_song_brief(app, "Stops on exhaustion")
    codex = FakeCodex(lambda stage, packet, calls: (
        {"accept": False, "revisions": ["Clarify the dramatic situation before composing"]}
        if stage == "brief" else None))
    producer = Producer(app, CodexCritic(codex), max_rounds=1, on_blocked="stop", seed=104)
    app.producer = producer
    assert producer.start()
    drive(app, producer, timeout=60)
    assert producer.phase == "failed" and producer.stage == "brief", producer.summary()
    assert producer.decisions == 0
    assert not app.project.melodies


def test_a_journal_that_cannot_be_written_stops_the_production(app, monkeypatch):
    """A record nobody can read is not a record: the production ends
    visibly rather than carrying on with its history lost."""
    a_whole_song_brief(app, "Unwritable journal")
    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    monkeypatch.setattr(ProjectState, "save",
                        lambda self, path: (_ for _ in ()).throw(OSError("disk gone")))
    drive(app, producer, timeout=120)
    assert producer.phase == "failed", producer.summary()
    assert "journal" in producer.error and "disk gone" in producer.error, producer.error
    assert not app.project.melodies, "nothing may be composed on an unrecorded review"


def test_the_voice_packet_measures_the_instrumental_sections(monkeypatch):
    """An instrumental section is *expected* silent; whether the take is
    silent there is a measurement, and the two are reported apart."""
    from types import SimpleNamespace
    import numpy as np
    from raagacomposer.production import producer as module

    prelude = SimpleNamespace(name="Prelude", start=0.0, end=1.0,
                              kind=SimpleNamespace(instrumental=True))
    pallavi = SimpleNamespace(name="Pallavi", start=1.0, end=2.0,
                              kind=SimpleNamespace(instrumental=False))
    take = SimpleNamespace(id="synthetic-take", kind="master", duration=2.0,
                           melody_version=1, lyrics_version=1)
    project = SimpleNamespace(vocal_master=take, latest_vocal=take,
                              melody=lambda: SimpleNamespace(sections=[prelude, pallavi]))
    monkeypatch.setattr(module.mastering, "report", lambda *args: "controlled")

    def packet_for(audio):
        fake_app = SimpleNamespace(project=project, sample_rate=8000,
                                   rendered=lambda kind: SimpleNamespace(audio=audio),
                                   current_voice=lambda: SimpleNamespace(name="test voice"))
        producer = Producer(fake_app, SimpleNamespace(available=False))
        monkeypatch.setattr(producer, "_packet", lambda stage, body: body, raising=False)
        return producer._evidence_voice()[0]

    # Sound in both sections: the Prelude is expected silent and is not.
    loud = packet_for(np.full(16000, 0.2, dtype=np.float32))
    assert loud["expected_silent_sections"] == ["Prelude"]
    prelude_row = loud["instrumental_sections"][0]
    assert prelude_row["name"] == "Prelude" and not prelude_row["measured_silent"]
    assert prelude_row["rms_db"] > module.SILENCE_DB
    assert "instrumental_sections_silent" not in loud

    # Silence in the Prelude, sound in the Pallavi: the measurement says so.
    audio = np.zeros(16000, dtype=np.float32)
    audio[8000:] = 0.2
    quiet = packet_for(audio)
    assert quiet["instrumental_sections"][0]["measured_silent"]
    assert quiet["sung_sections"][0]["rms_db"] > module.SILENCE_DB


def _first_tune_with_a_locked_pallavi(app, title):
    from dataclasses import asdict
    a_whole_song_brief(app, title)
    app.generate_tune(seed=37)
    deadline = time.time() + 60
    while app.project.melody() is None and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    original = app.project.melody()
    assert original is not None, app.status_text
    while app.jobs.active_jobs() and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    pallavi = next(s for s in original.sections if s.kind == SectionKind.PALLAVI)
    app.set_section_lock(pallavi.id, True)
    notes = [asdict(n) for n in original.notes if n.section_id == pallavi.id]
    return original, pallavi, asdict(pallavi), notes


def _pallavi_unchanged(app, original, protected, notes):
    from dataclasses import asdict
    current = app.project.melody()
    assert current is original, "the active tune was replaced"
    pallavi = next(s for s in current.sections if s.kind == SectionKind.PALLAVI)
    assert pallavi.locked and asdict(pallavi) == protected
    assert [asdict(n) for n in current.notes if n.section_id == pallavi.id] == notes


def test_a_lock_placed_during_a_review_is_composed_around(app):
    """The creator locks a section while the Critic thinks about the brief.
    The tune stage sees the lock when it comes to compose and keeps that
    section, rather than stopping (it used to stop: Arya's
    producer_late_lock_review_cases, before composing around locks existed)."""
    a_whole_song_brief(app, "Late lock")
    app.generate_tune(seed=37)
    deadline = time.time() + 60
    while (app.project.melody() is None or app.jobs.active_jobs()) and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    original = app.project.melody()
    pallavi = next(s for s in original.sections if s.kind == SectionKind.PALLAVI)
    gate = GateCodex()
    app.critic = CodexCritic(gate)
    producer = app.produce_song(seed=104)
    try:
        pump_until(app, gate.entered.is_set)
        assert producer.stage == "brief" and producer.phase == "reviewing"
        app.set_section_lock(pallavi.id, True)
        from dataclasses import asdict
        protected = asdict(pallavi)
        notes = [asdict(n) for n in original.notes if n.section_id == pallavi.id]
        gate.release.set()
        drive(app, producer)
        assert producer.phase == "done", producer.report()
        assert len(app.project.melodies) == 2
        current = app.project.melody()
        kept = next(s for s in current.sections if s.kind == SectionKind.PALLAVI)
        assert kept.locked and _notes_of(current, kept.id) == notes, "the late lock was not kept"
        assert "Pallavi" in "\n".join(producer.events)
    finally:
        gate.release.set()
        if not producer.finished:
            producer.cancel("test over")


def test_a_lock_placed_while_the_composer_works_is_kept_when_its_result_lands(app, monkeypatch):
    """The last door: the lock arrives after the tune stage was submitted
    and before the composer's result lands (Arya's
    producer_generation_lock_review_cases).  The ticket generate_tune
    writes now carries every section that was unlocked at submission, so
    _tune_ready refuses the result rather than replace a locked section."""
    import raagacomposer.app as controller_module
    a_whole_song_brief(app, "Lock during composition")
    app.generate_tune(seed=37)
    deadline = time.time() + 60
    while (app.project.melody() is None or app.jobs.active_jobs()) and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    original = app.project.melody()
    pallavi = next(s for s in original.sections if s.kind == SectionKind.PALLAVI)
    entered, release = threading.Event(), threading.Event()
    real_generate = controller_module.melody_engine.generate

    def held_generation(*args, **kwargs):
        entered.set()
        assert release.wait(30), "the test must release the composer"
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(controller_module.melody_engine, "generate", held_generation)
    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    try:
        pump_until(app, entered.is_set)
        assert producer.stage == "tune" and producer.phase == "working"
        app.set_section_lock(pallavi.id, True)
        from dataclasses import asdict
        protected = asdict(pallavi)
        notes = [asdict(n) for n in original.notes if n.section_id == pallavi.id]
        release.set()
        drive(app, producer, timeout=60)
        assert producer.phase == "failed" and "locked" in producer.error, producer.error
        assert len(app.project.melodies) == 1, "the composer's result was kept over a lock"
        _pallavi_unchanged(app, original, protected, notes)
    finally:
        release.set()
        if not producer.finished:
            producer.cancel("test over")


# ----------------------------------------------------------------------
# Composing around locked sections (queue item 3, 2026-09-08)
# ----------------------------------------------------------------------
def _notes_of(melody, section_id):
    from dataclasses import asdict
    return [asdict(n) for n in melody.notes if n.section_id == section_id]


def test_a_production_composes_around_a_locked_pallavi(app):
    """A locked section is kept - notes, timing, words - and the rest of the
    song is made around it; the report says which sections were kept."""
    original, pallavi, protected, notes = _first_tune_with_a_locked_pallavi(app, "Around the lock")
    # Words first, so the Pallavi has a line to keep.
    app.set_section_lock(pallavi.id, False)
    app.providers.llm = None
    app.generate_lyrics(seed=4)
    deadline = time.time() + 60
    while app.jobs.active_jobs() and time.time() < deadline:
        app.pump()
        time.sleep(0.02)
    words = app.project.lyrics_version()
    kept_line = next(l for l in words.lines if l.section_id == pallavi.id)
    app.set_lyric_line_lock(kept_line.id, True)
    app.set_section_lock(pallavi.id, True)
    kept_text = kept_line.text
    others = [s for s in original.sections if s.id != pallavi.id and not s.kind.instrumental]

    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    drive(app, producer)
    assert producer.phase == "done", producer.report()

    current = app.project.melody()
    assert current is not original and current.version == original.version + 1
    kept = next(s for s in current.sections if s.kind == SectionKind.PALLAVI)
    assert kept.locked and _notes_of(current, kept.id) == notes, "the locked Pallavi changed"
    assert any(_notes_of(current, s.id) != _notes_of(original, s.id) for s in others), \
        "nothing outside the lock was composed"
    latest = app.project.lyrics_version()
    line = next(l for l in latest.lines if l.section_id == kept.id)
    assert line.text == kept_text and line.locked, "the locked words changed"
    report = producer.report()
    assert "Pallavi" in report and "locked" in report.lower()
    assert "kept locked" in "\n".join(producer.events).lower()


def test_a_fully_locked_tune_is_explained_not_composed(app):
    original, *_ = _first_tune_with_a_locked_pallavi(app, "All locked")
    for section in original.sections:
        app.set_section_lock(section.id, True)
    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    drive(app, producer, timeout=60)
    assert producer.phase == "failed" and "every section is locked" in producer.error, producer.error
    assert len(app.project.melodies) == 1 and app.project.melody() is original


def test_a_lock_placed_while_the_variation_runs_refuses_its_result(app, monkeypatch):
    import raagacomposer.app as controller_module
    original, pallavi, protected, notes = _first_tune_with_a_locked_pallavi(app, "Lock during variation")
    other = next(s for s in original.sections if s.id != pallavi.id and not s.kind.instrumental)
    entered, release = threading.Event(), threading.Event()
    real_variation = controller_module.melody_engine.variation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30), "the test must release the composer"
        return real_variation(*args, **kwargs)

    monkeypatch.setattr(controller_module.melody_engine, "variation", held)
    app.critic = CodexCritic(FakeCodex())
    producer = app.produce_song(seed=104)
    try:
        pump_until(app, entered.is_set)
        app.set_section_lock(other.id, True)
        release.set()
        drive(app, producer, timeout=60)
        assert producer.phase == "failed" and "locked" in producer.error, producer.error
        assert len(app.project.melodies) == 1
    finally:
        release.set()
        if not producer.finished:
            producer.cancel("test over")
