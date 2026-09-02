"""Claude language-model adapter (optional).

Enabled when the ``anthropic`` package is installed and a key is available
from the environment or the credentials file.  Nothing is hard-coded and the
application works fully without it.

One instance is one model.  The router holds two of them - a strong model for
the work whose quality is heard, and a cheap fast one for the work that merely
has to be right - because "route on complexity and cost" is not a decision an
adapter can make for itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from . import prompts, tasks
from .base import LLMProvider

log = get_logger("providers.claude")


@dataclass(frozen=True)
class ModelSpec:
    """What we need to know about a model to use it correctly and rank it.

    ``effort`` marks the models that accept ``output_config.effort`` and
    adaptive thinking.  Sending either to a model that predates them is a 400,
    so this is a correctness flag, not a preference.  Prices are USD per
    million tokens; ``cost`` blends them at three input tokens per output.
    """

    strength: int
    input_usd: float
    output_usd: float
    effort: bool

    @property
    def cost(self) -> float:
        return round((3 * self.input_usd + self.output_usd) / 4, 2)


MODELS: Dict[str, ModelSpec] = {
    "claude-opus-5":    ModelSpec(95, 5.00, 25.00, effort=True),
    "claude-sonnet-5":  ModelSpec(85, 2.00, 10.00, effort=True),
    "claude-haiku-4-5": ModelSpec(65, 1.00, 5.00, effort=False),
}

DEFAULT_MODEL = "claude-opus-5"          # the work whose quality is heard
DEFAULT_LIGHT_MODEL = "claude-haiku-4-5"  # the work that must merely be right
UNKNOWN = ModelSpec(75, 3.00, 15.00, effort=False)


class ClaudeLLM(LLMProvider):
    """One Claude model behind the standard provider interface."""

    requires_key = True

    def __init__(self, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 effort: str = "medium",
                 thinking: bool = True) -> None:
        self.model = model
        self.spec = MODELS.get(model, UNKNOWN)
        self.effort = effort
        self.thinking = thinking
        self.name = f"claude:{model}"
        self.strength = self.spec.strength
        self.cost_per_mtok = self.spec.cost
        self._client = None
        self._error = ""
        key = api_key or Settings.secret("anthropic_api_key")
        if not key:
            self._error = "No API key configured."
            return
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:                                 # noqa: BLE001
            self._error = f"anthropic package unavailable: {exc}"
            log.info(self._error)

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> str:
        if not self.available:
            return self._error
        return f"ready ({self.model}, ~${self.spec.cost}/Mtok)"

    # -- plumbing ----------------------------------------------------------
    def _ask(self, system: str, prompt: str, task: tasks.TaskSpec) -> str:
        if self._client is None:
            raise RuntimeError(self._error or "Claude provider unavailable")
        max_tokens = task.max_tokens
        kwargs: Dict[str, Any] = {}
        if self.spec.effort:
            kwargs["output_config"] = {"effort": self.effort}
            # Thinking earns its keep on the two tasks with real constraints to
            # satisfy; elsewhere it only adds latency.  When it is on, the
            # answer and the reasoning share max_tokens, so the ceiling has to
            # rise or a long think truncates the reply we actually wanted.
            if self.thinking and task.quality_critical:
                kwargs["thinking"] = {"type": "adaptive"}
                max_tokens = max(max_tokens * 4, 8000)
        response = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}], **kwargs)
        if getattr(response, "stop_reason", "") == "refusal":
            detail = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"declined by the model ({getattr(detail, 'category', 'unknown')})")
        parts = []
        for block in response.content:
            if getattr(block, "type", "") != "text":
                continue                       # skip thinking blocks
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    def _json(self, system: str, prompt: str, task: tasks.TaskSpec) -> Any:
        return prompts.extract_json(self._ask(system, prompt, task))

    # -- capabilities ------------------------------------------------------
    def write_lyrics(self, slots: Sequence[Any], brief: Any) -> List[str]:
        task = tasks.TASKS[tasks.WRITE_LYRICS]
        system, user = prompts.lyrics(slots, brief)
        return prompts.as_lyrics(self._json(system, user, task))

    def classify_intent(self, text: str, intents: Sequence[str]) -> Dict[str, Any]:
        task = tasks.TASKS[tasks.CLASSIFY_INTENT]
        system, user = prompts.intent(text, intents)
        return prompts.as_intent(self._json(system, user, task))

    def suggest_raagas(self, brief: Any, candidates: Sequence[str]
                       ) -> List[Dict[str, str]]:
        task = tasks.TASKS[tasks.SUGGEST_RAAGAS]
        system, user = prompts.raagas(brief, candidates)
        return prompts.as_raagas(self._json(system, user, task))

    def suggest_instruments(self, description: str,
                            catalog: Sequence[str]) -> List[str]:
        task = tasks.TASKS[tasks.SUGGEST_INSTRUMENTS]
        system, user = prompts.instruments(description, catalog)
        return prompts.as_instruments(self._json(system, user, task), catalog)

    def explain(self, question: str, context: str = "") -> str:
        task = tasks.TASKS[tasks.EXPLAIN]
        system, user = prompts.explain(question, context)
        return self._ask(system, user, task)
