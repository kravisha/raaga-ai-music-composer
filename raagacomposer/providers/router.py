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
from . import tasks
from .base import LLMProvider

log = get_logger("providers.router")

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
                 quality_floor: int = MEDIUM_FLOOR) -> None:
        self._factories = list(factories)
        self.policy = policy if policy in POLICIES else "auto"
        self.refresh_seconds = max(0.0, refresh_seconds)
        self.quality_floor = quality_floor
        self._backends: List[LLMProvider] = []
        self._checked = 0.0
        self._key_seen = bool(Settings.secret("anthropic_api_key"))
        self.last_route: Dict[str, str] = {}
        self.refresh()

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
        if spec.quality_critical:
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
        elif self.policy == "claude_first":
            ordered = sorted(ordered, key=lambda b: 1 if b.is_local else 0)
        return ordered

    @staticmethod
    def _empty(result: Any) -> bool:
        return result is None or result == [] or result == {} or result == ""

    def _call(self, task_name: str, run: Callable[[LLMProvider], Any],
              default: Any) -> Any:
        self._ensure_fresh()
        chain = self.chain(task_name)
        for backend in chain:
            try:
                result = run(backend)
            except Exception as exc:                             # noqa: BLE001
                # Offline, rate limited, refused, a model that went away: the
                # task moves down the chain rather than being lost.
                log.warning("%s failed on %s: %s", backend.name, task_name, exc)
                continue
            if self._empty(result):
                log.info("%s returned nothing for %s", backend.name, task_name)
                continue
            self.last_route[task_name] = backend.name
            return result
        if chain:
            log.info("no backend answered %s - using the built-in engine",
                     task_name)
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
    def write_lyrics(self, slots: Sequence[Any], brief: Any) -> List[str]:
        return self._call(tasks.WRITE_LYRICS,
                          lambda b: b.write_lyrics(slots, brief), [])

    def classify_intent(self, text: str, intents: Sequence[str]) -> Dict[str, Any]:
        return self._call(tasks.CLASSIFY_INTENT,
                          lambda b: b.classify_intent(text, intents), {})

    def suggest_raagas(self, brief: Any, candidates: Sequence[str]
                       ) -> List[Dict[str, str]]:
        return self._call(tasks.SUGGEST_RAAGAS,
                          lambda b: b.suggest_raagas(brief, candidates), [])

    def suggest_instruments(self, description: str,
                            catalog: Sequence[str]) -> List[str]:
        return self._call(tasks.SUGGEST_INSTRUMENTS,
                          lambda b: b.suggest_instruments(description, catalog),
                          [])

    def explain(self, question: str, context: str = "") -> str:
        return self._call(tasks.EXPLAIN,
                          lambda b: b.explain(question, context), "")
