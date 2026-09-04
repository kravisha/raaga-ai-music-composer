"""The provider manager: which model answers which question.

The application asks for five different things (``tasks.py``) and may have
several backends able to answer (``claude_llm.py``, ``local_llm.py``).  This
module is the one place that decides between them, so no caller ever names a
model and no adapter ever decides it is the right one for a job.

Three things order the choice, in the order the creator asked for them:

* **complexity** - a hard task goes to the strongest backend that is up; an
  easy one goes to the cheapest that can still do it.
* **cost** - a local model costs nothing, so it wins every tie, and the cheap
  Claude model is preferred over the expensive one wherever the task allows.
* **offline availability** - a backend that is not reachable is not in the
  running, and one that fails mid-request hands the task straight to the next
  in line rather than losing it.

Below all of them sits the floor that was always there: if no backend answers,
the method returns empty and the caller falls back to the built-in rule and
lexicon engines.  That is why the application still works with nothing
installed and no key.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from . import escalation, tasks
from .base import LLMProvider

log = get_logger("providers.router")


def _short(value: Any, limit: int = 600) -> str:
    """An answer as the routing log should keep it: readable, bounded."""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


Factory = Callable[[], LLMProvider]

#: A backend weaker than this is not a first choice for a middling task; it
#: stays in the chain as the offline fallback.
MEDIUM_FLOOR = 50

POLICIES = ("auto", "local_first", "claude_first", "local_only",
            "claude_only", "off")


class RoutedLLM(LLMProvider):
    """One :class:`LLMProvider` face over several backends."""

    name = "router"
    requires_key = False

    def __init__(self, factories: Sequence[Factory], policy: str = "auto",
                 refresh_seconds: float = 30.0,
                 quality_floor: int = MEDIUM_FLOOR,
                 thresholds: Optional[escalation.Thresholds] = None,
                 attempt_log: Optional[escalation.AttemptLog] = None,
                 tiers: Optional[Dict[str, str]] = None,
                 order: Optional[Sequence[str]] = None) -> None:
        self._factories = list(factories)
        self.policy = policy if policy in POLICIES else "auto"
        self.refresh_seconds = max(0.0, refresh_seconds)
        self.quality_floor = quality_floor
        self.thresholds = thresholds or escalation.Thresholds()
        self.attempt_log = attempt_log
        #: model name -> tier name, so an attempt can say which rung it was.
        self.tier_of = {model: tier for tier, model in (tiers or {}).items()}
        #: The rungs, cheapest first.  This is the escalation order the config
        #: states, and in the judged modes it is what orders the local
        #: backends - not their strength.  Every local model costs nothing, so
        #: the cost key cannot separate them, and ordering by strength put the
        #: largest and slowest first, which is the opposite of "a cheap first
        #: attempt" and made the configured order decorative.
        self.order = list(order or [])
        self._backends: List[LLMProvider] = []
        self._checked = 0.0
        self._key_seen = bool(Settings.secret("anthropic_api_key"))
        self.last_route: Dict[str, str] = {}
        #: The mode and the model behind the most recent answer for each
        #: task.  Recorded because a quality dip has to be attributable:
        #: without it, runs made under different modes are not comparable
        #: and a change we made ourselves looks like the model getting worse.
        self.last_decision: Dict[str, escalation.Decision] = {}
        self.refresh()

    @property
    def judged(self) -> bool:
        """Whether answers are judged and escalated rather than merely tried.

        The judged loop is what makes "attempt local first" safe without a
        strength floor, so the two go together: in the older modes there is
        no judge, and the floor stays.
        """
        return self.policy in ("local_first", "local_only")

    # -- backends ----------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild every backend and re-probe what is reachable.

        Called at startup and whenever the creator changes a setting.  This is
        the expensive check - it can touch the network - so the hot path uses
        :meth:`_ensure_fresh` instead.
        """
        built: List[LLMProvider] = []
        for factory in self._factories:
            try:
                backend = factory()
            except Exception as exc:                             # noqa: BLE001
                log.warning("provider failed to load: %s", exc)
                continue
            built.append(backend)
            log.info("provider %s: %s", backend.name, backend.status())
        self._backends = built
        self._checked = time.monotonic()
        self._key_seen = bool(Settings.secret("anthropic_api_key"))

    def _ensure_fresh(self) -> None:
        """Cheap re-check, safe to call before every request.

        Only one thing is worth watching continuously: whether a key has
        appeared since we last looked.  Reading it is a file or environment
        lookup, not a network call, so this stays out of the way of a spoken
        instruction that has to be understood before the creator stops
        talking.  When it changes, the backends are rebuilt - which is what
        lets Claude be switched on by adding a key, with the application
        already running and no code touched.
        """
        if self.refresh_seconds <= 0:
            return
        now = time.monotonic()
        if now - self._checked < self.refresh_seconds:
            return
        self._checked = now
        if bool(Settings.secret("anthropic_api_key")) != self._key_seen:
            log.info("API key state changed - rebuilding providers")
            self.refresh()

    @property
    def backends(self) -> List[LLMProvider]:
        return list(self._backends)

    @property
    def available(self) -> bool:
        self._ensure_fresh()
        return any(b.available for b in self._rank_all())

    def status(self) -> str:
        allowed = self._rank_all()
        if allowed:
            return (f"{len(allowed)} backend(s), policy={self.policy}: "
                    + ", ".join(b.name for b in allowed))
        # A backend that is running but ruled out by the policy is a different
        # situation from having none at all, and saying so saves the creator
        # hunting for a fault that is a setting.
        up = [b for b in self._backends if b.available]
        if up:
            return (f"policy={self.policy} excludes the {len(up)} backend(s) "
                    f"that are up - rule and lexicon engines are in use")
        return ("no language model configured - rule and lexicon engines "
                "are in use")

    def _rank_all(self) -> List[LLMProvider]:
        """Backends the current policy permits, regardless of task."""
        ready = [b for b in self._backends if b.available]
        if self.policy == "off":
            return []
        if self.policy == "local_only":
            return [b for b in ready if b.is_local]
        if self.policy == "claude_only":
            return [b for b in ready if not b.is_local]
        return ready

    # -- routing -----------------------------------------------------------
    def chain(self, task_name: str) -> List[LLMProvider]:
        """The backends to try for ``task_name``, best first."""
        ready = self._rank_all()
        if not ready:
            return []
        spec = tasks.spec(task_name)

        # Measured, not assumed: llama3.2:3b took 704 seconds over ten lyric
        # lines on a CPU here and returned nothing usable, while the built-in
        # lexicon engine fits all ten exactly and instantly.  For the two
        # tasks whose quality is heard, a backend below the floor is not a
        # slow answer - it is a long wait for a worse one - so it is excluded
        # rather than merely ranked last.  Raise llm_local_strength when the
        # local model is genuinely bigger.
        # The strength floor used to exclude weak backends from the two
        # quality-critical tasks outright, on a measurement taken once.  The
        # standing policy is that nothing is excluded before it has been
        # tried: a local model is attempted whatever its declared strength,
        # and the judge decides afterwards on what it actually produced.  The
        # floor stays for the older modes, where there is no judge to catch a
        # bad answer and a strength number is all there is to go on.
        if spec.quality_critical and not self.judged:
            ready = [b for b in ready if b.strength >= self.quality_floor]
            if not ready:
                return []

        if spec.latency_critical:
            # Reached mid-sentence: a local answer now beats a better one after
            # a network round trip.
            key = lambda b: (0 if b.is_local else 1, b.cost_per_mtok, -b.strength)
        elif spec.complexity is tasks.Complexity.HIGH:
            key = lambda b: (-b.strength, b.cost_per_mtok)
        elif spec.complexity is tasks.Complexity.LOW:
            key = lambda b: (b.cost_per_mtok, -b.strength)
        else:
            key = lambda b: (0 if b.strength >= MEDIUM_FLOOR else 1,
                             b.cost_per_mtok, -b.strength)
        ordered = sorted(ready, key=key)

        # A stated preference reorders the groups but keeps the task's own
        # ordering inside each - sorting here is stable.
        if self.policy == "local_first":
            ordered = sorted(ordered, key=lambda b: 0 if b.is_local else 1)
        if self.judged and self.order:
            # ... and inside the local group, the configured escalation order
            # decides, so the cheap first attempt really is attempted first.
            ordered = sorted(ordered, key=self._rung)
        elif self.policy == "claude_first":
            ordered = sorted(ordered, key=lambda b: 1 if b.is_local else 0)
        return ordered

    def _rung(self, backend: LLMProvider) -> tuple:
        """Where a backend sits in the configured escalation order.

        Paid backends stay after every local one; a local backend the config
        does not name goes after the ones it does, rather than jumping the
        queue on a strength number the policy no longer uses.
        """
        if not backend.is_local:
            return (2, len(self.order), backend.name)
        tier = self.tier_of.get(getattr(backend, "model", ""), "")
        if tier in self.order:
            return (0, self.order.index(tier), backend.name)
        return (1, len(self.order), backend.name)

    @staticmethod
    def _empty(result: Any) -> bool:
        return result is None or result == [] or result == {} or result == ""

    def _call(self, task_name: str, run: Callable[[LLMProvider], Any],
              default: Any, validate: Optional[Callable[[Any], bool]] = None,
              prompt: str = "") -> Any:
        """Ask the chain for an answer, judging each one before accepting it.

        In the judged modes this is the standing policy's attempt-then-
        escalate loop: try the cheapest local backend, judge what comes back
        on schema, then log-probabilities, then a second sample, and move on
        only when it has actually been found wanting.  In the older modes the
        judge has no thresholds worth applying, so an answer is accepted if
        it is not empty - exactly as before.
        """
        self._ensure_fresh()
        chain = self.chain(task_name)
        if not chain:
            return default

        # What each backend actually said, kept for the log: the policy wants
        # the local output and the paid output side by side, or a threshold
        # cannot be tuned against a real case.
        outputs: Dict[str, Any] = {}

        def ask(backend: LLMProvider) -> escalation.Sample:
            value = run(backend)
            outputs[backend.name] = _short(value)
            return escalation.Sample(
                value=value,
                mean_logprob=getattr(backend, "last_mean_logprob", None),
                seconds=float(getattr(backend, "last_seconds", 0.0) or 0.0))

        decision = escalation.escalate(
            chain, ask, self.thresholds, mode=self.policy,
            validate=validate if self.judged else None,
            name_of=lambda b: b.name,
            tier_of=lambda b: self.tier_of.get(
                getattr(b, "model", ""), "paid" if not b.is_local else ""),
            is_paid=lambda b: not b.is_local,
        )
        self.last_decision[task_name] = decision
        if self.attempt_log is not None and decision.attempts:
            self.attempt_log.write(task_name, prompt, decision, outputs)

        if decision.answered:
            self.last_route[task_name] = decision.backend
            log.info("%s answered %s %s", decision.backend, task_name,
                     decision.summary())
            return decision.value
        log.info("no backend answered %s - using the built-in engine (%s)",
                 task_name, decision.summary())
        return default

    def explain_routing(self) -> str:
        """A readable account of where each task would go right now."""
        rows = []
        for name in (tasks.WRITE_LYRICS, tasks.SUGGEST_RAAGAS, tasks.EXPLAIN,
                     tasks.SUGGEST_INSTRUMENTS, tasks.CLASSIFY_INTENT):
            spec = tasks.spec(name)
            chain = self.chain(name)
            route = " -> ".join(b.name for b in chain) or "built-in engine"
            rows.append(f"{name:20s} {spec.complexity.value:6s} {route}")
        return "\n".join(rows)

    # -- capabilities ------------------------------------------------------
    # Each passes a validator: signal one of the judge, and the only one that
    # is certain rather than a threshold.  It earns its place immediately -
    # qwen3:4b answered a raaga request at a confident -0.53 mean logprob
    # with one of its three entries keyed ``": "`` instead of ``"raaga"``.
    # Nothing but a schema check catches that.
    def write_lyrics(self, slots: Sequence[Any], brief: Any) -> List[str]:
        wanted = len(list(slots))

        def valid(lines: Any) -> bool:
            return (isinstance(lines, list) and bool(lines)
                    and all(isinstance(line, str) and line.strip()
                            for line in lines)
                    and (not wanted or len(lines) == wanted))

        return self._call(tasks.WRITE_LYRICS,
                          lambda b: b.write_lyrics(slots, brief), [],
                          validate=valid, prompt=str(getattr(brief, "feel", "")))

    def classify_intent(self, text: str, intents: Sequence[str]) -> Dict[str, Any]:
        allowed = set(intents)

        def valid(answer: Any) -> bool:
            return (isinstance(answer, dict)
                    and bool(str(answer.get("intent", "")).strip())
                    and (not allowed or answer.get("intent") in allowed
                         or answer.get("intent") == "unknown"))

        return self._call(tasks.CLASSIFY_INTENT,
                          lambda b: b.classify_intent(text, intents), {},
                          validate=valid, prompt=text)

    def suggest_raagas(self, brief: Any, candidates: Sequence[str]
                       ) -> List[Dict[str, str]]:
        known = {c.lower() for c in candidates}

        def valid(rows: Any) -> bool:
            if not isinstance(rows, list) or not rows:
                return False
            for row in rows:
                if not isinstance(row, dict):
                    return False
                name = str(row.get("raaga", "")).strip().lower()
                # Only from the supplied list, and every row has to name one.
                if not name or (known and name not in known):
                    return False
            return True

        return self._call(tasks.SUGGEST_RAAGAS,
                          lambda b: b.suggest_raagas(brief, candidates), [],
                          validate=valid,
                          prompt=f"{getattr(brief, 'mood', '')} / "
                                 f"{getattr(brief, 'feel', '')}")

    def suggest_instruments(self, description: str,
                            catalog: Sequence[str]) -> List[str]:
        known = {c.lower() for c in catalog}

        def valid(picks: Any) -> bool:
            return (isinstance(picks, list) and bool(picks)
                    and all(isinstance(p, str) and (not known
                                                    or p.lower() in known)
                            for p in picks))

        return self._call(tasks.SUGGEST_INSTRUMENTS,
                          lambda b: b.suggest_instruments(description, catalog),
                          [], validate=valid, prompt=description)

    def explain(self, question: str, context: str = "") -> str:
        return self._call(tasks.EXPLAIN,
                          lambda b: b.explain(question, context), "",
                          validate=lambda t: isinstance(t, str) and bool(t.strip()),
                          prompt=question)
