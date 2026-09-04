"""The attempt-then-escalate loop and its judge.

The standing routing policy: attempt local first whatever the declared
complexity, judge the answer on schema then log-probabilities then a second
sample, and reach a paid model only on a judged failure.  All of it is
exercised with fakes here - the loop is given callables on purpose, so what
it does can be pinned down without spending a model call or a penny.
"""
from __future__ import annotations

import json

import pytest

from raagacomposer.providers import escalation
from raagacomposer.providers.escalation import (ACCEPTED, DIVERGED, EMPTY,
                                                ERROR, LOW_CONFIDENCE, SCHEMA,
                                                TIMEOUT, Attempt, AttemptLog,
                                                Sample, Thresholds, escalate,
                                                judge)

pytestmark = pytest.mark.unit

T = Thresholds(logprob_floor=-1.10, divergence=0.15, attempt_seconds=90.0)


def vector(**over):
    base = {d: 0.5 for d in ("sadness", "warmth", "joy")}
    base.update(over)
    return base


def in_range(value) -> bool:
    return all(0.0 <= v <= 1.0 for v in value.values())


# -- the judge, signal by signal -------------------------------------------
def test_an_empty_answer_fails_before_anything_else():
    assert judge(Sample(value=None), T).verdict == EMPTY
    assert judge(Sample(value={}), T).verdict == EMPTY
    assert judge(Sample(value=""), T).verdict == EMPTY


def test_schema_is_checked_first_and_costs_nothing():
    """A malformed answer is rejected without a second sample being paid for."""
    resamples = []
    attempt = judge(Sample(value=vector(sadness=9.0), mean_logprob=-0.01), T,
                    validate=in_range,
                    resample=lambda: resamples.append(1) or Sample())
    assert attempt.verdict == SCHEMA
    assert not resamples, "a schema failure must not pay for a second sample"


def test_a_validator_that_raises_is_a_schema_failure_not_a_crash():
    def explode(_):
        raise ValueError("no dimension called that")

    attempt = judge(Sample(value=vector()), T, validate=explode)
    assert attempt.verdict == SCHEMA
    assert "no dimension called that" in attempt.detail


def test_a_confident_answer_is_taken_at_its_word():
    attempt = judge(Sample(value=vector(), mean_logprob=-0.05), T,
                    validate=in_range,
                    resample=lambda: pytest.fail("should not resample"))
    assert attempt.verdict == ACCEPTED
    assert not attempt.resampled


def test_a_mean_below_the_floor_is_low_confidence():
    attempt = judge(Sample(value=vector(), mean_logprob=-2.0), T,
                    validate=in_range)
    assert attempt.verdict == LOW_CONFIDENCE
    assert "-2.000" in attempt.detail and "-1.100" in attempt.detail


def test_a_borderline_mean_is_settled_by_a_second_sample():
    """Between the floor and comfortably above it, ask again."""
    borderline = Sample(value=vector(), mean_logprob=-1.0)
    agreeing = judge(borderline, T, validate=in_range,
                     resample=lambda: Sample(value=vector()))
    assert agreeing.verdict == ACCEPTED
    assert agreeing.resampled

    diverging = judge(borderline, T, validate=in_range,
                      resample=lambda: Sample(value=vector(joy=0.99,
                                                           sadness=0.01)))
    assert diverging.verdict == DIVERGED
    assert "tolerance" in diverging.detail


def test_a_runtime_that_cannot_report_logprobs_is_not_failed_for_it():
    """``None`` means "cannot say", never "bad" - llama.cpp through this
    application does not report them, and must still be usable."""
    silent = Sample(value=vector(), mean_logprob=None)
    assert judge(silent, T, validate=in_range,
                 resample=lambda: Sample(value=vector())).verdict == ACCEPTED
    assert judge(silent, T, validate=in_range).verdict == ACCEPTED


def test_divergence_is_zero_for_the_same_answer_and_one_for_nothing():
    same = Sample(value=vector())
    assert escalation.divergence(same, Sample(value=vector())) == 0.0
    assert escalation.divergence(same, Sample(value=None, text="")) == 1.0


# -- the loop ---------------------------------------------------------------
def _chain(*names):
    return list(names)


def test_the_first_good_local_answer_wins_and_nothing_paid_is_reached():
    asked = []

    def ask(name):
        asked.append(name)
        return Sample(value=vector(), mean_logprob=-0.05)

    decision = escalate(_chain("small", "mid", "claude"), ask, T,
                        validate=in_range, is_paid=lambda n: n == "claude")
    assert decision.value == vector()
    assert decision.backend == "small"
    assert asked == ["small"], "nothing beyond the first should be asked"
    assert not decision.paid
    assert not decision.escalated


def test_a_judged_failure_escalates_one_step_at_a_time():
    def ask(name):
        if name == "small":
            return Sample(value=vector(sadness=9.0), mean_logprob=-0.05)
        if name == "mid":
            return Sample(value=vector(), mean_logprob=-3.0)
        return Sample(value=vector(), mean_logprob=-0.05)

    decision = escalate(_chain("small", "mid", "claude"), ask, T,
                        validate=in_range, is_paid=lambda n: n == "claude")
    assert [a.verdict for a in decision.attempts] == [SCHEMA, LOW_CONFIDENCE,
                                                      ACCEPTED]
    assert decision.backend == "claude"
    assert decision.paid and decision.escalated


def test_a_backend_that_raises_is_a_failure_like_any_other():
    def ask(name):
        if name == "small":
            raise RuntimeError("ollama went away")
        return Sample(value=vector(), mean_logprob=-0.05)

    decision = escalate(_chain("small", "mid"), ask, T, validate=in_range)
    assert decision.attempts[0].verdict == ERROR
    assert "ollama went away" in decision.attempts[0].detail
    assert decision.backend == "mid"


def test_a_slow_answer_fails_even_when_it_is_a_good_one():
    """The measured case: ten lyric lines took 704 seconds here and the
    judge as stated would have accepted them."""
    def ask(name):
        if name == "small":
            return Sample(value=vector(), mean_logprob=-0.01, seconds=704.0)
        return Sample(value=vector(), mean_logprob=-0.05, seconds=2.0)

    decision = escalate(_chain("small", "mid"), ask, T, validate=in_range)
    assert decision.attempts[0].verdict == TIMEOUT
    assert "704s" in decision.attempts[0].detail
    assert decision.backend == "mid"


def test_nothing_answering_leaves_a_decision_with_no_value():
    """The caller falls back to its built-in engine; the loop does not
    invent an answer to avoid saying so."""
    decision = escalate(_chain("small", "mid"), lambda n: Sample(value=None),
                        T, validate=in_range)
    assert decision.value is None
    assert not decision.answered
    assert len(decision.attempts) == 2


def test_the_loop_asks_only_for_what_it_was_given():
    """paid_only leaves the local candidates out rather than running them
    and discarding the result, so a rollback costs nothing in latency."""
    asked = []
    escalate(_chain("claude"), lambda n: asked.append(n) or Sample(
        value=vector(), mean_logprob=-0.05), T, validate=in_range)
    assert asked == ["claude"]


def test_every_attempt_is_reported_as_it_happens():
    seen = []
    escalate(_chain("small", "mid"),
             lambda n: Sample(value=vector(), mean_logprob=(
                 -3.0 if n == "small" else -0.05)),
             T, validate=in_range,
             on_attempt=lambda a, c: seen.append((a.backend, a.verdict)))
    assert seen == [("small", LOW_CONFIDENCE), ("mid", ACCEPTED)]


# -- the log ----------------------------------------------------------------
def test_the_log_records_what_the_thresholds_need_to_be_tuned_against(tmp_path):
    def ask(name):
        if name == "small":
            return Sample(value=vector(sadness=9.0), mean_logprob=-0.05)
        return Sample(value=vector(), mean_logprob=-0.05)

    decision = escalate(_chain("small", "claude"), ask, T, validate=in_range,
                        mode="local_first", is_paid=lambda n: n == "claude")
    log = AttemptLog(tmp_path / "routing_attempts.jsonl")
    log.write("suggest_raagas", "sad, lonely, warm", decision,
              outputs={"small": "bad", "claude": "good"})

    rows = [json.loads(line) for line in
            (tmp_path / "routing_attempts.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "local_first"           # the mode it ran under
    assert row["answered_by"] == "claude"         # the model that answered
    assert row["escalated"] is True
    assert row["prompt"] == "sad, lonely, warm"   # the brief
    assert [a["verdict"] for a in row["attempts"]] == [SCHEMA, ACCEPTED]
    assert row["outputs"] == {"small": "bad", "claude": "good"}


def test_a_log_that_cannot_be_written_never_breaks_the_answer(tmp_path):
    log = AttemptLog(tmp_path / "not-a-dir" / "x.jsonl")
    log.path = tmp_path                      # a directory: writing must fail
    log.write("t", "p", escalation.Decision())     # must not raise


def test_thresholds_come_from_the_one_config_block():
    class FakeSettings:
        routing_logprob_floor = -2.5
        routing_divergence = 0.4
        routing_sample_temperature = 0.9
        routing_attempt_seconds = 30.0

    t = Thresholds.from_settings(FakeSettings())
    assert (t.logprob_floor, t.divergence, t.sample_temperature,
            t.attempt_seconds) == (-2.5, 0.4, 0.9, 30.0)


def test_numbers_are_compared_as_numbers_not_as_text():
    """Two affect vectors that contradict each other on every value are
    textually almost identical - keys and punctuation are most of the
    string - so a text-only measure would call them the same answer."""
    one = Sample(value={"sadness": 0.9, "joy": 0.1, "warmth": 0.8})
    other = Sample(value={"sadness": 0.1, "joy": 0.9, "warmth": 0.2})
    assert escalation.divergence(one, other) > 0.5

    text_only = 1.0 - __import__("difflib").SequenceMatcher(
        None, one.key(), other.key()).ratio()
    assert text_only < 0.15, "the text measure really is blind to this"

    close = Sample(value={"sadness": 0.9, "joy": 0.1, "warmth": 0.75})
    assert escalation.divergence(one, close) < 0.1


def test_a_missing_dimension_counts_against_agreement():
    full = Sample(value={"sadness": 0.9, "joy": 0.1})
    partial = Sample(value={"sadness": 0.9})
    assert escalation.divergence(full, partial) > 0.0
