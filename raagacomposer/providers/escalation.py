"""Attempt locally, judge the answer, escalate only if it fails.

The standing routing policy, in one place.  Every model-driven step tries a
local model first - whatever its declared complexity, and without any
strength floor deciding in advance that it is not worth asking - and a paid
model is reached only when the local answer has actually been found wanting.

The judge applies three signals, in this order, stopping at the first verdict:

1. **Schema validity.**  Malformed or out-of-range output is a failure.  This
   is the only one that is pass/fail rather than a threshold, and it is first
   because it is free and certain.
2. **Token log-probabilities**, where the runtime exposes them.  A mean below
   the configured floor is low confidence.  Ollama returns these; llama.cpp
   through this application does not, and a backend that cannot say goes to
   the next signal rather than being failed for staying silent.
3. **Two samples at non-zero temperature.**  Only when log-probabilities are
   unavailable or borderline, because it costs a second generation.  Answers
   that diverge beyond tolerance are a failure.

A deadline sits across all of it.  The judge as the policy states it measures
quality, not time, and a local model that takes ten minutes to produce a good
answer has still failed the creator - the measured case in ``DECISIONS.md``
is ten lyric lines taking 704 seconds on this machine.  Exceeding
``attempt_seconds`` is therefore a failure like any other, and escalates.

Nothing here knows what a raaga is, and nothing here imports a backend: the
loop is given callables and judges what comes back, so it can be tested
without a model and reused for any task.
"""
from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from ..core.logging_setup import get_logger

log = get_logger("providers.escalation")

#: How close to the floor still counts as borderline.  A mean comfortably
#: above the floor is taken at its word; within this fraction of it, a second
#: sample is worth paying for.  Floors are negative log-probabilities, so
#: this scales towards zero.
BORDERLINE = 0.75

#: Verdicts a judged attempt can end in.  ``ACCEPTED`` is the only one that
#: stops the loop; the rest name why it did not.
ACCEPTED = "accepted"
SCHEMA = "schema"
LOW_CONFIDENCE = "low_confidence"
DIVERGED = "diverged"
EMPTY = "empty"
ERROR = "error"
TIMEOUT = "timeout"


@dataclass(frozen=True)
class Thresholds:
    """The judge's numbers, from one config block so they can be tuned."""

    logprob_floor: float = -1.10
    divergence: float = 0.15
    sample_temperature: float = 0.5
    attempt_seconds: float = 90.0

    @classmethod
    def from_settings(cls, settings: Any) -> "Thresholds":
        return cls(
            logprob_floor=float(getattr(settings, "routing_logprob_floor",
                                        cls.logprob_floor)),
            divergence=float(getattr(settings, "routing_divergence",
                                     cls.divergence)),
            sample_temperature=float(getattr(
                settings, "routing_sample_temperature", cls.sample_temperature)),
            attempt_seconds=float(getattr(settings, "routing_attempt_seconds",
                                          cls.attempt_seconds)),
        )


@dataclass
class Sample:
    """One generation, and whatever the runtime was willing to say about it."""

    value: Any = None
    #: Mean token log-probability, or ``None`` where the runtime does not
    #: expose them.  ``None`` is "cannot say", never "bad".
    mean_logprob: Optional[float] = None
    seconds: float = 0.0
    text: str = ""

    def key(self) -> str:
        """A stable string for comparing two samples of the same request."""
        if self.text:
            return self.text
        if self.value is None:
            return ""
        try:
            return json.dumps(self.value, sort_keys=True, default=str)
        except Exception:                                        # noqa: BLE001
            return str(self.value)


@dataclass
class Attempt:
    """What one backend was asked, what it said, and how it was judged."""

    backend: str = ""
    tier: str = ""
    paid: bool = False
    verdict: str = ""
    detail: str = ""
    mean_logprob: Optional[float] = None
    seconds: float = 0.0
    resampled: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict == ACCEPTED


@dataclass
class Decision:
    """The answer, and the full account of how it was reached."""

    value: Any = None
    mode: str = ""
    backend: str = ""
    tier: str = ""
    paid: bool = False
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].accepted

    @property
    def escalated(self) -> bool:
        return len([a for a in self.attempts if not a.accepted]) > 0

    def summary(self) -> str:
        route = " -> ".join(
            f"{a.backend}:{a.verdict}" for a in self.attempts) or "nothing tried"
        return f"[{self.mode}] {route}"


def _numeric_map(value: Any) -> Optional[dict]:
    """``{key: float}`` if that is what this is, else ``None``."""
    if not isinstance(value, dict) or not value:
        return None
    out = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        out[str(key)] = float(item)
    return out


def divergence(first: Sample, second: Sample) -> float:
    """How far apart two answers to the same request are, 0 (same) to 1.

    Numbers are compared as numbers.  Text similarity on its own is nearly
    blind to exactly the disagreement that matters here: two fourteen-
    dimension affect vectors that contradict each other on every value are
    textually almost identical, because the keys, the braces and the commas
    are most of the string.  For a mapping of numbers - which is what the
    schema-constrained tasks return - the measure is the mean absolute
    difference across the union of keys, so a model that says 0.9 once and
    0.1 the next time is caught.  Anything else falls back to text.
    """
    left_map, right_map = _numeric_map(first.value), _numeric_map(second.value)
    if left_map is not None and right_map is not None:
        keys = set(left_map) | set(right_map)
        if not keys:
            return 0.0
        total = sum(abs(left_map.get(k, 0.0) - right_map.get(k, 0.0))
                    for k in keys)
        return min(1.0, total / len(keys))

    left, right = first.key(), second.key()
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, left, right).ratio()


def judge(sample: Sample, thresholds: Thresholds,
          validate: Optional[Callable[[Any], bool]] = None,
          resample: Optional[Callable[[], Sample]] = None) -> Attempt:
    """The three signals, in order, stopping at the first verdict."""
    attempt = Attempt(mean_logprob=sample.mean_logprob, seconds=sample.seconds)

    # 1. Schema validity.  Free, certain, and first.
    if sample.value is None or sample.value == [] or sample.value == {} \
            or sample.value == "":
        attempt.verdict = EMPTY
        attempt.detail = "no answer"
        return attempt
    if validate is not None:
        try:
            ok = bool(validate(sample.value))
        except Exception as exc:                                 # noqa: BLE001
            ok, exc_text = False, f"{exc.__class__.__name__}: {exc}"
            attempt.detail = exc_text
        if not ok:
            attempt.verdict = SCHEMA
            attempt.detail = attempt.detail or "did not validate"
            return attempt

    # 2. Log-probabilities, where the runtime exposes them.
    if sample.mean_logprob is not None:
        if sample.mean_logprob < thresholds.logprob_floor:
            attempt.verdict = LOW_CONFIDENCE
            attempt.detail = (f"mean logprob {sample.mean_logprob:.3f} below "
                              f"{thresholds.logprob_floor:.3f}")
            return attempt
        # Comfortably above the floor: no need to pay for a second sample.
        if sample.mean_logprob > thresholds.logprob_floor * BORDERLINE:
            attempt.verdict = ACCEPTED
            return attempt

    # 3. Borderline, or the runtime cannot say: ask twice and compare.
    if resample is not None:
        second = resample()
        attempt.resampled = True
        spread = divergence(sample, second)
        if spread > thresholds.divergence:
            attempt.verdict = DIVERGED
            attempt.detail = (f"two samples differed by {spread:.2f}, "
                              f"tolerance {thresholds.divergence:.2f}")
            return attempt

    attempt.verdict = ACCEPTED
    return attempt


def escalate(candidates: Sequence[Any], ask: Callable[[Any], Sample],
             thresholds: Thresholds, mode: str = "local_first",
             validate: Optional[Callable[[Any], bool]] = None,
             resample: Optional[Callable[[Any], Sample]] = None,
             name_of: Callable[[Any], str] = str,
             tier_of: Callable[[Any], str] = lambda c: "",
             is_paid: Callable[[Any], bool] = lambda c: False,
             on_attempt: Optional[Callable[[Attempt, Any], None]] = None,
             ) -> Decision:
    """Try each candidate in turn until one is judged good enough.

    ``candidates`` are already in the order the mode wants them.  In
    ``claude_only`` the caller has left the local ones out entirely rather
    than passing them here to be skipped, so that a rollback costs nothing in
    latency; this function does not silently drop anything it is given.
    """
    decision = Decision(mode=mode)
    for candidate in candidates:
        started = time.monotonic()
        try:
            sample = ask(candidate)
        except Exception as exc:                                 # noqa: BLE001
            attempt = Attempt(verdict=ERROR,
                              detail=f"{exc.__class__.__name__}: {exc}",
                              seconds=time.monotonic() - started)
        else:
            elapsed = sample.seconds or (time.monotonic() - started)
            if elapsed > thresholds.attempt_seconds:
                attempt = Attempt(
                    verdict=TIMEOUT, seconds=elapsed,
                    mean_logprob=sample.mean_logprob,
                    detail=(f"took {elapsed:.0f}s, over the "
                            f"{thresholds.attempt_seconds:.0f}s deadline"))
            else:
                again = (lambda c=candidate: resample(c)) if resample else None
                attempt = judge(sample, thresholds, validate, again)
                attempt.seconds = elapsed

        attempt.backend = name_of(candidate)
        attempt.tier = tier_of(candidate)
        attempt.paid = is_paid(candidate)
        decision.attempts.append(attempt)
        if on_attempt is not None:
            on_attempt(attempt, candidate)

        if attempt.accepted:
            decision.value = sample.value
            decision.backend = attempt.backend
            decision.tier = attempt.tier
            decision.paid = attempt.paid
            return decision
        log.info("%s did not answer well enough (%s: %s)",
                 attempt.backend, attempt.verdict, attempt.detail)
    return decision


class AttemptLog:
    """Every attempt, so the thresholds can be tuned against real cases.

    The policy is explicit that this is the point: without the brief, the
    local output, the failing signal and the paid output side by side, a
    threshold is a guess that cannot be improved.  One JSON object per line,
    appended, never rewritten.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write(self, task: str, prompt: str, decision: Decision,
              outputs: Optional[dict] = None) -> None:
        row = {
            "at": time.time(),
            "task": task,
            "mode": decision.mode,
            "answered_by": decision.backend or "built-in engine",
            "tier": decision.tier,
            "paid": decision.paid,
            "escalated": decision.escalated,
            "prompt": prompt[:2000],
            "attempts": [
                {"backend": a.backend, "tier": a.tier, "paid": a.paid,
                 "verdict": a.verdict, "detail": a.detail,
                 "mean_logprob": a.mean_logprob,
                 "seconds": round(a.seconds, 3), "resampled": a.resampled}
                for a in decision.attempts
            ],
            "outputs": outputs or {},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:                                 # noqa: BLE001
            # Never let bookkeeping break the answer the creator is waiting for.
            log.warning("could not write the routing log: %s", exc)
