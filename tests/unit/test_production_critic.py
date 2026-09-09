import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from raagacomposer.production.critic import (CancelledReview, CodexCliTransport,
                                           CodexCritic, ReviewValidationError,
                                           Verdict, review_stage)
from raagacomposer.production.state import ProjectState


class FakeCodex:
    """An explicitly fake transport for tests; no real model is called."""
    provider = "codex"
    model = "test-double"

    def __init__(self, change=None):
        self.change = change
        self.calls = []

    def __call__(self, prompt, schema, *, cancelled=None):
        packet = json.loads(prompt.split("Evidence packet:\n", 1)[1])
        self.calls.append(packet)
        response = dict(request_id=packet["request_id"], stage=packet["stage"],
                        accept=True, blocked=False, reason="The supplied score supports this stage",
                        revisions=[], strengths=["Preserve the returning phrase"],
                        lessons=["Use a recurring phrase to connect sections"],
                        uncertainty=["Diction has not been heard"], evidence=["score:v1"])
        if self.change:
            self.change(response)
        return response


class DoneProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


class WaitingProcess:
    """A child that can exit on terminate or require kill, without real waiting."""
    def __init__(self, ignore_terminate=False, on_wait=lambda: None):
        self.returncode = None
        self.ignore_terminate = ignore_terminate
        self.on_wait = on_wait
        self.terminated = False
        self.killed = False
        self.reaped = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.on_wait()
            raise subprocess.TimeoutExpired("fake-child", timeout)
        self.reaped = True
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = -1

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_guiding_review_retains_strengths_lessons_uncertainty_and_source():
    backend = FakeCodex()
    state = ProjectState(project_id="song", brief={"mood": "hopeful"})
    verdict = review_stage(CodexCritic(backend), "tune", state,
                           {"notes": [60, 64, 67], "evidence_ref": "score:v1"}, artifact_ref="tune-v1")
    assert verdict.accept and verdict.strengths and verdict.lessons and verdict.uncertainty
    assert verdict.provider == "codex" and verdict.model == "test-double"
    assert state.is_accepted("tune", "tune-v1")
    assert state.records[-1].evidence == ["score:v1"]


@pytest.mark.parametrize("change", [
    lambda x: x.update(request_id="an-old-song"),
    lambda x: x.update(stage="lyrics"),
    lambda x: x.update(accept="true"),
    lambda x: x.update(revisions=["Change these notes"]),
    lambda x: x.update(blocked=True),
    lambda x: x.pop("uncertainty"),
    lambda x: x.update(execute="run a command"),
])
def test_bad_or_mismatched_model_response_never_accepts(change):
    state = ProjectState(project_id="song")
    result = review_stage(CodexCritic(FakeCodex(change)), "tune", state, {}, artifact_ref="tune-v1")
    assert result.blocked and not result.accept
    assert not state.is_accepted("tune", "tune-v1")


def test_review_exhaustion_stops_calls_and_never_becomes_acceptance():
    def revise(value):
        value.update(accept=False, revisions=["Shorten the repeated phrase; leave room for the singer"])
    backend = FakeCodex(revise)
    critic = CodexCritic(backend)
    state = ProjectState(project_id="song")
    for attempt in (1, 2):
        result = review_stage(critic, "tune", state, {}, artifact_ref=f"tune-v{attempt}",
                              round_number=attempt, max_rounds=2)
        assert not result.accept and not result.blocked
    exhausted = review_stage(critic, "tune", state, {}, artifact_ref="tune-v3",
                             round_number=3, max_rounds=2)
    assert exhausted.blocked and len(backend.calls) == 2
    assert [record.verdict for record in state.records] == ["revise", "revise", "blocked"]


def test_repeating_round_one_cannot_bypass_the_budget():
    backend = FakeCodex(lambda x: x.update(accept=False, revisions=["More variation"] ))
    state = ProjectState(project_id="song")
    for _ in range(3):
        review_stage(CodexCritic(backend), "tune", state, {}, artifact_ref="v1", max_rounds=1)
    assert len(backend.calls) == 1


def test_creator_change_during_review_rejects_result():
    state = ProjectState(project_id="song", brief={"mood": "joy"})
    def changed(_):
        state.brief["mood"] = "grief"
    verdict = review_stage(CodexCritic(FakeCodex(changed)), "tune", state, {}, artifact_ref="v1")
    assert verdict.blocked and "stale" in verdict.reason
    assert not state.is_accepted("tune", "v1")


def test_stale_review_is_recorded_against_old_revision_and_does_not_spend_new_budget():
    state = ProjectState(project_id="song")
    def changed(_):
        state.invalidate_from("tune", "Creator requested a different tune")
    verdict = review_stage(CodexCritic(FakeCodex(changed)), "tune", state, {}, artifact_ref="v1")
    assert verdict.blocked and state.records[-1].revision == 0 and state.revision == 1
    current = review_stage(CodexCritic(FakeCodex()), "tune", state, {}, artifact_ref="v2", max_rounds=1)
    assert current.accept and state.is_accepted("tune", "v2")


def test_guidance_survives_save_reopen_for_specialist_followups(tmp_path):
    state = ProjectState(project_id="song")
    review_stage(CodexCritic(FakeCodex()), "tune", state, {}, artifact_ref="v1")
    reopened = ProjectState.load(state.save(tmp_path / "production.json"))
    record = reopened.records[-1]
    assert record.strengths == ["Preserve the returning phrase"]
    assert record.lessons == ["Use a recurring phrase to connect sections"]
    assert record.uncertainty == ["Diction has not been heard"]
    assert record.model == "test-double"


def test_local_reviewer_cannot_replace_codex():
    class Local:
        def review(self, *args):
            return Verdict(True, "Notes in range", provider="local rules")
    state = ProjectState(project_id="song")
    verdict = review_stage(Local(), "tune", state, {}, artifact_ref="v1")
    assert verdict.blocked and verdict.provider == "local rules"
    with pytest.raises(ValueError):
        CodexCritic(Local())


def test_failure_does_not_echo_private_exception_payload():
    class Broken:
        def review(self, *args):
            raise RuntimeError("private-token-should-not-appear")
    state = ProjectState(project_id="song")
    verdict = review_stage(Broken(), "tune", state, {}, artifact_ref="v1")
    assert verdict.blocked and "private-token" not in verdict.reason
    assert "private-token" not in state.records[-1].rationale


def test_startup_availability_does_not_call_a_model(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: None)
    called = []
    reviewer = CodexCritic(CodexCliTransport(tmp_path, popen=lambda *a, **k: called.append(1)))
    assert not reviewer.available
    assert "unavailable" in reviewer.status()
    assert called == [] and not list(tmp_path.iterdir())


def test_cli_adapter_uses_stdin_read_only_and_ephemeral_workspace(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    observed = []
    def fake_popen(command, **kwargs):
        prompt = kwargs["stdin"].read()
        kwargs["observed_prompt"] = prompt
        observed.append((command, kwargs))
        answer = Path(command[command.index("--output-last-message") + 1])
        assert Path(command[command.index("--output-schema") + 1]).is_file()
        packet = json.loads(prompt.split("Evidence packet:\n", 1)[1])
        response = FakeCodex()(prompt, {})
        assert response["request_id"] == packet["request_id"]
        answer.write_text(json.dumps(response), encoding="utf-8")
        return DoneProcess()
    backend = CodexCliTransport(tmp_path, executable="codex.exe", popen=fake_popen)
    assert not observed  # Startup never generates a hidden model request.
    result = CodexCritic(backend).review("tune", ProjectState(project_id="song"),
                                       {"words": "quotes and $() stay data"})
    assert result.accept
    command, kwargs = observed[0]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command and command[-1] == "-"
    assert not any("bypass" in arg or "ignore-rules" in arg for arg in command)
    assert kwargs["shell"] is False and kwargs["stderr"] == subprocess.DEVNULL
    assert "quotes and $()" in kwargs["observed_prompt"] and "quotes and $()" not in " ".join(command)
    assert kwargs["stdin"].closed
    assert not list(tmp_path.iterdir())


def test_explicit_executable_outside_path_is_used_without_installing(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    binary = tmp_path / "desktop install" / "codex.exe"
    binary.parent.mkdir()
    binary.write_bytes(b"test double, never executed")
    binary.chmod(0o700)
    monkeypatch.setattr(module.shutil, "which", lambda executable: None)
    observed = []
    def fake_popen(command, **kwargs):
        observed.append(command)
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(FakeCodex()(kwargs["stdin"].read(), {})))
        return DoneProcess()
    transport = CodexCliTransport(tmp_path / "work", executable=binary, popen=fake_popen)
    assert transport.available and not observed
    assert CodexCritic(transport).review("brief", ProjectState(project_id="song"), {}).accept
    assert observed[0][0] == str(binary.resolve())
    assert binary.read_bytes() == b"test double, never executed"


def test_missing_explicit_executable_never_falls_back_to_path(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: "a-different-codex.exe")
    calls = []
    transport = CodexCliTransport(tmp_path, executable=tmp_path / "missing.exe",
                                 popen=lambda *a, **k: calls.append(1))
    assert not transport.available
    song = ProjectState(project_id="song")
    verdict = review_stage(CodexCritic(transport), "brief", song, {}, artifact_ref="brief-v1")
    assert verdict.blocked and not calls and not song.is_accepted("brief", "brief-v1")


def test_path_resolution_is_fixed_before_changing_working_directory(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: "relative-bin/codex.exe")
    transport = CodexCliTransport(tmp_path)
    assert transport.resolved_executable == str(Path("relative-bin/codex.exe").resolve())


def test_long_song_review_sends_selected_context_and_preserves_complete_history(tmp_path):
    from raagacomposer.production.state import PRODUCTION_STAGES, StageRecord, canonical_json
    song = ProjectState(project_id="long-lived-song", revision=60,
                        feedback=[f"Creator revision {i}: change the ending" for i in range(60)])
    for revision in range(60):
        for stage in PRODUCTION_STAGES:
            for attempt in (1, 2, 3):
                song.records.append(StageRecord(stage=stage, role="critic", artifact_ref=f"{stage}:{revision}",
                    rationale="Keep the opening motif while making the transition more spacious. " * 4,
                    verdict="accept" if attempt == 3 else "revise", round=attempt,
                    revision=revision, provider="codex", strengths=["Clear recurring motif"],
                    lessons=["Preserve identity while varying the answer phrase"]))
    song.add_record(StageRecord(stage="tune", role="critic", artifact_ref="tune:60",
                    rationale="Give the singer more space", verdict="revise", round=1,
                    revision=60, provider="codex", revisions=["Shorten the repeated phrase"]))
    original = song.to_dict()
    assert len(canonical_json(original).encode("utf-8")) > 256_000
    backend = FakeCodex()
    assert CodexCritic(backend).review("tune", song, {"score": "new version"}).accept
    packet = backend.calls[0]
    assert len(canonical_json(packet).encode("utf-8")) < 20_000
    assert len(packet["state"]["records"]) == 8
    assert {r["revision"] for r in packet["state"]["records"]} == {59, 60}
    assert packet["state"]["records"][-1]["revisions"] == ["Shorten the repeated phrase"]
    assert packet["context_selection"]["omitted_records"] == len(song.records) - 8
    assert packet["context_selection"]["omitted_feedback"] == 28
    assert packet["state"]["feedback"] == song.feedback[-32:]
    assert song.to_dict() == original and not song.is_accepted("tune", "tune:59")
    assert ProjectState.load(song.save(tmp_path / "production.json")).to_dict() == original


def test_selected_history_never_promotes_local_acceptance_or_drops_current_feedback():
    from raagacomposer.production.state import StageRecord
    song = ProjectState(project_id="song", revision=2, feedback=["Use the latest words"])
    for revision, provider, verdict in ((0, "codex", "accept"), (1, "local rules", "accept"),
                                       (1, "codex", "revise"), (2, "codex", "blocked")):
        song.records.append(StageRecord(stage="lyrics", role="critic", artifact_ref=f"v{revision}",
            rationale="Reference evidence", verdict=verdict, round=1, revision=revision, provider=provider))
    backend = FakeCodex()
    CodexCritic(backend).review("lyrics", song, {})
    selected = backend.calls[0]["state"]
    assert [r["revision"] for r in selected["records"]] == [0, 2]
    assert selected["feedback"] == ["Use the latest words"]


def test_oversized_current_evidence_still_blocks_before_model_request():
    backend = FakeCodex()
    state = ProjectState(project_id="song")
    result = review_stage(CodexCritic(backend), "tune", state,
                          {"lyrics": "அ" * 100_000}, artifact_ref="v1")
    assert result.blocked and not backend.calls and not state.is_accepted("tune", "v1")


@pytest.mark.parametrize("mode", ["nonzero", "missing", "malformed", "spawn_error"])
def test_cli_failures_block_without_fallback(tmp_path, monkeypatch, mode):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    def fake_popen(command, **kwargs):
        if mode == "spawn_error":
            raise OSError("private-child-detail")
        if mode == "malformed":
            Path(command[command.index("--output-last-message") + 1]).write_text("not-json")
        return DoneProcess(returncode=1 if mode == "nonzero" else 0)
    state = ProjectState(project_id="song")
    result = review_stage(CodexCritic(CodexCliTransport(tmp_path, popen=fake_popen)),
                          "tune", state, {}, artifact_ref="v1")
    assert result.blocked and not state.is_accepted("tune", "v1")
    assert not list(tmp_path.iterdir())


def test_precancelled_review_never_spawns_or_spends_budget(tmp_path):
    called = []
    state = ProjectState(project_id="song")
    critic = CodexCritic(CodexCliTransport(tmp_path, popen=lambda *a, **k: called.append(1)))
    with pytest.raises(CancelledReview):
        review_stage(critic, "brief", state, {}, artifact_ref="brief-v1", cancelled=lambda: True)
    assert not called and not state.records and not list(tmp_path.iterdir())


@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_inflight_cancel_terminates_and_reaps_child_without_record(tmp_path, monkeypatch, ignore_terminate):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    event = threading.Event()
    child = WaitingProcess(ignore_terminate=ignore_terminate, on_wait=event.set)
    critic = CodexCritic(CodexCliTransport(tmp_path, popen=lambda *a, **k: child))
    state = ProjectState(project_id="song")
    with pytest.raises(CancelledReview):
        review_stage(critic, "brief", state, {}, artifact_ref="brief-v1", cancelled=event.is_set)
    assert child.terminated and child.reaped and child.killed == ignore_terminate
    assert not state.records and not list(tmp_path.iterdir())


def test_timeout_reaps_child_and_blocks_without_private_output(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    def advance():
        clock[0] += 0.5
    child = WaitingProcess(ignore_terminate=True, on_wait=advance)
    state = ProjectState(project_id="song")
    verdict = review_stage(CodexCritic(CodexCliTransport(tmp_path, timeout=1,
        popen=lambda *a, **k: child)), "brief", state, {}, artifact_ref="brief-v1")
    assert verdict.blocked and "TimeoutExpired" in verdict.reason
    assert child.terminated and child.killed and child.reaped
    assert len(state.records) == 1 and not state.is_accepted("brief", "brief-v1")
    assert not list(tmp_path.iterdir())


def test_cancel_on_return_discards_valid_verdict_and_allows_round_one_again():
    event = threading.Event()
    backend = FakeCodex(lambda response: event.set())
    state = ProjectState(project_id="song")
    with pytest.raises(CancelledReview):
        review_stage(CodexCritic(backend), "brief", state, {}, artifact_ref="v1", cancelled=event.is_set)
    assert len(backend.calls) == 1 and not state.records
    verdict = review_stage(CodexCritic(FakeCodex()), "brief", state, {}, artifact_ref="v1")
    assert verdict.accept and state.records[-1].round == 1


def test_cancellation_callback_failure_still_stops_owned_child(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    spawned = []
    child = WaitingProcess()
    def spawn(*args, **kwargs):
        spawned.append(child)
        return child
    def cancelled():
        if spawned:
            raise RuntimeError("private-callback-error")
        return False
    state = ProjectState(project_id="song")
    # Transport owns cleanup even if a badly behaved caller's callback raises.
    with pytest.raises(RuntimeError, match="private-callback-error"):
        CodexCliTransport(tmp_path, popen=spawn)("test prompt", {}, cancelled=cancelled)
    assert child.terminated and child.reaped and not list(tmp_path.iterdir())
    assert not state.records


def test_real_harmless_child_is_cancelled_and_reaped_with_large_stdin(tmp_path):
    """No model: exercise Windows process/stdin/cleanup using only a sleeping Python child."""
    children = []
    event = threading.Event()
    timer = threading.Timer(0.25, event.set)
    def spawn(command, **kwargs):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(process)
        timer.start()
        return process
    try:
        with pytest.raises(CancelledReview):
            CodexCliTransport(tmp_path, executable=sys.executable, popen=spawn)(
                "அ" * 100_000, {}, cancelled=event.is_set)
        # Observe cleanup before the test's own emergency cleanup can mask it.
        assert len(children) == 1 and children[0].poll() is not None
        assert not list(tmp_path.iterdir())
    finally:
        timer.cancel()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)


@pytest.mark.parametrize("change,diagnostic", [
    (lambda x: x.update(revisions=["private-model-wording"]),
     "A blocked or revision-required artifact cannot be accepted"),
    (lambda x: x.pop("uncertainty"), "Incomplete or unexpected review fields"),
    (lambda x: x.update(request_id="private-wrong-request"),
     "Review belongs to a different stage or artifact snapshot"),
    (lambda x: x.update(accept="private-not-a-boolean"), "Verdict flags must be booleans"),
    (lambda x: x.update(evidence="private-not-an-array"), "evidence must contain nonempty text values"),
])
def test_owned_validation_explains_failure_without_echoing_response(change, diagnostic):
    state = ProjectState(project_id="song")
    verdict = review_stage(CodexCritic(FakeCodex(change)), "brief", state, {}, artifact_ref="v1")
    assert verdict.blocked and diagnostic in verdict.reason
    assert "private-" not in verdict.reason
    assert state.records[-1].rationale == verdict.reason
    assert not state.is_accepted("brief", "v1")


@pytest.mark.parametrize("typed", [False, True])
def test_external_errors_cannot_smuggle_payload_in_validation_messages(typed):
    class Broken:
        def review(self, *args):
            if typed:
                exc = ReviewValidationError("contradictory_acceptance")
                exc.args = ("private-token",)
                exc.code = "private-token"
                raise exc
            raise ValueError("A blocked or revision-required artifact cannot be accepted: private-token")
    state = ProjectState(project_id="song")
    verdict = review_stage(Broken(), "brief", state, {}, artifact_ref="v1")
    assert verdict.blocked and "private-token" not in verdict.reason
    if typed:
        assert "Invalid critic response" in verdict.reason
    else:
        assert "ValueError" in verdict.reason


def test_prior_stage_context_is_slim_but_current_rounds_and_saved_journal_stay_complete(tmp_path):
    from raagacomposer.production.state import StageRecord, canonical_json
    state = ProjectState(project_id="song", revision=1, feedback=["Keep the gentle opening"])
    for stage, revision, provider in (("tune", 0, "codex"), ("brief", 1, "codex"),
                                     ("tune", 1, "local rules"), ("lyrics", 1, "codex"),
                                     ("voice", 1, "codex")):
        state.records.append(StageRecord(stage=stage, role="critic", provider=provider,
            artifact_ref=f"{stage}-v{revision}", revision=revision, round=1,
            rationale="Keep the phrase; change the ending", verdict="revise" if stage == "voice" else "accept",
            revisions=["Leave space at the cadence"], evidence=["Evidence " * 1000],
            strengths=["Strength " * 1000], lessons=["Lesson " * 1000], uncertainty=["Unknown " * 1000]))
    path = state.save(tmp_path / "production.json")
    saved_bytes = path.read_bytes()
    original = state.to_dict()
    backend = FakeCodex()
    assert CodexCritic(backend).review("voice", state, {}).accept
    packet = backend.calls[0]
    records = packet["state"]["records"]
    assert packet["context_selection"]["summarized_records"] == 4
    assert packet["context_selection"]["omitted_records"] == 0
    assert packet["context_selection"]["complete_current_stage_rounds"] is True
    assert records[-1] == original["records"][-1]
    for compressed, full in zip(records[:-1], original["records"][:-1]):
        assert all(compressed[key] == full[key] for key in
                   ("stage", "artifact_ref", "revision", "round", "provider", "role", "rationale", "verdict", "revisions"))
        assert all(key not in compressed for key in ("evidence", "strengths", "lessons", "uncertainty"))
    assert records[2]["provider"] == "local rules"  # Never erase source when summarizing acceptance.
    assert len(canonical_json(packet)) < len(canonical_json(original)) / 2
    assert state.to_dict() == original and path.read_bytes() == saved_bytes
    assert ProjectState.load(path).to_dict() == original


def test_windows_exit_race_preserves_cancellation_and_reaps_child(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    event = threading.Event()
    class ExitingChild(WaitingProcess):
        def terminate(self):
            self.returncode = 0
            raise PermissionError("child exited in Windows race")
    child = ExitingChild(on_wait=event.set)
    with pytest.raises(CancelledReview):
        CodexCliTransport(tmp_path, popen=lambda *a, **k: child)("test", {}, cancelled=event.is_set)
    assert child.reaped and not list(tmp_path.iterdir())
