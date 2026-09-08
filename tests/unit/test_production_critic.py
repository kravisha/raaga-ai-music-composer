import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from raagacomposer.production.critic import CodexCliTransport, CodexCritic, Verdict, review_stage
from raagacomposer.production.state import ProjectState


class FakeCodex:
    """An explicitly fake transport for tests; no real model is called."""
    provider = "codex"
    model = "test-double"

    def __init__(self, change=None):
        self.change = change
        self.calls = []

    def __call__(self, prompt, schema):
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
    reviewer = CodexCritic(CodexCliTransport(tmp_path, run=lambda *a, **k: called.append(1)))
    assert not reviewer.available
    assert "unavailable" in reviewer.status()
    assert called == [] and not list(tmp_path.iterdir())


def test_cli_adapter_uses_stdin_read_only_and_ephemeral_workspace(tmp_path, monkeypatch):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    observed = []
    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        answer = Path(command[command.index("--output-last-message") + 1])
        assert Path(command[command.index("--output-schema") + 1]).is_file()
        packet = json.loads(kwargs["input"].split("Evidence packet:\n", 1)[1])
        response = FakeCodex()(kwargs["input"], {})
        assert response["request_id"] == packet["request_id"]
        answer.write_text(json.dumps(response), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    backend = CodexCliTransport(tmp_path, executable="codex.exe", run=fake_run)
    assert not observed  # Startup never generates a hidden model request.
    result = CodexCritic(backend).review("tune", ProjectState(project_id="song"),
                                       {"words": "quotes and $() stay data"})
    assert result.accept
    command, kwargs = observed[0]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command and command[-1] == "-"
    assert not any("bypass" in arg or "ignore-rules" in arg for arg in command)
    assert kwargs["shell"] is False and kwargs["timeout"] == 120
    assert "quotes and $()" in kwargs["input"] and "quotes and $()" not in " ".join(command)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["timeout", "nonzero", "missing", "malformed"])
def test_cli_failures_block_without_fallback(tmp_path, monkeypatch, mode):
    from raagacomposer.production import critic as module
    monkeypatch.setattr(module.shutil, "which", lambda executable: executable)
    def fake_run(command, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if mode == "malformed":
            Path(command[command.index("--output-last-message") + 1]).write_text("not-json")
        return SimpleNamespace(returncode=1 if mode == "nonzero" else 0)
    state = ProjectState(project_id="song")
    result = review_stage(CodexCritic(CodexCliTransport(tmp_path, run=fake_run)),
                          "tune", state, {}, artifact_ref="v1")
    assert result.blocked and not state.is_accepted("tune", "v1")
    assert not list(tmp_path.iterdir())
