"""Lightweight local language models (optional).

Two runtimes, both off by default and neither bundled:

* **Ollama** - a server already running on this machine, spoken to over
  ``127.0.0.1``.  Nothing leaves the machine and there is no key.
* **llama.cpp** - a GGUF file under ``<config>/models/llm/``, loaded in
  process by ``llama-cpp-python``.  No server at all.

Both are small-model runtimes on purpose.  They exist so the application keeps
its intelligence when there is no network and no key, and so the cheap
high-volume work does not have to travel.  Neither is asked to do what it
cannot: the router sends them the easy tasks first and the hard ones only when
nothing better is up.

Nothing here downloads a model.  If no model is present the adapter reports
itself unavailable and names the one command that would fix it, in the same
way the speech backends do.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from . import prompts, tasks
from .base import LLMProvider

log = get_logger("providers.local_llm")


def _same_model(have: str, wanted: str) -> bool:
    """Is ``have`` the model ``wanted`` asks for?

    A bare name means ``:latest`` on both sides, so "llama3" and
    "llama3:latest" are one model.  The tag otherwise has to match: this used
    to compare only the part before the colon, which was harmless while one
    Ollama model was ever configured and wrong the moment two tags of one
    family were - ``qwen3:8b`` reported itself ready because ``qwen3:4b`` had
    been pulled, and then answered every request with a 404.
    """
    have_name, _, have_tag = (have or "").partition(":")
    want_name, _, want_tag = (wanted or "").partition(":")
    if not have_name or have_name != want_name:
        return False
    return (have_tag or "latest") == (want_tag or "latest")


def _mean_logprob(entries: Any) -> Optional[float]:
    """Mean token log-probability, or ``None`` if the runtime said nothing.

    Ollama returns ``[{"token": ..., "logprob": ...}, ...]``.  A runtime
    that omits the field is reporting that it cannot say, which the
    routing judge treats differently from a low score.
    """
    if not entries:
        return None
    values = [float(e["logprob"]) for e in entries
              if isinstance(e, dict) and isinstance(e.get("logprob"), (int, float))]
    return sum(values) / len(values) if values else None


DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
PROBE_TIMEOUT = 1.5


def models_dir() -> Path:
    p = config_dir() / "models" / "llm"
    p.mkdir(parents=True, exist_ok=True)
    return p


class SmallModelLLM(LLMProvider):
    """What the two local runtimes have in common.

    The prompts, the parsing and all five capabilities live here; a subclass
    supplies only :meth:`_ask` - how a question actually reaches a model - and
    its own availability. Neither runtime inherits from the other, so neither
    depends on attributes the other happens to set.
    """

    is_local = True
    cost_per_mtok = 0.0
    requires_key = False

    _error = ""
    _ok = False

    #: What the last call reported about itself, for the routing judge.
    #: ``None`` for the mean means the runtime does not expose token
    #: log-probabilities - which is not the same as reporting a bad one.
    last_mean_logprob: Optional[float] = None
    last_seconds: float = 0.0

    @property
    def available(self) -> bool:
        return self._ok

    def _ask(self, system: str, prompt: str, task: tasks.TaskSpec) -> str:
        raise NotImplementedError

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


class OllamaLLM(SmallModelLLM):
    """A small model served by Ollama on this machine."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT,
                 model: str = DEFAULT_OLLAMA_MODEL,
                 strength: int = 40, timeout: float = 60.0) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or DEFAULT_OLLAMA_MODEL
        self.strength = strength
        self.timeout = timeout
        self.name = f"ollama:{self.model}"
        self._error = ""
        self._ok = False
        self._probe()

    # -- availability ------------------------------------------------------
    def _probe(self) -> None:
        """Ask the server what it has.  One call, at construction."""
        try:
            with urllib.request.urlopen(f"{self.endpoint}/api/tags",
                                        timeout=PROBE_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                                 # noqa: BLE001
            self._error = (f"no Ollama server at {self.endpoint} "
                           f"({exc.__class__.__name__})")
            return
        have = [str(m.get("name", "")) for m in data.get("models", [])]
        if any(_same_model(h, self.model) for h in have):
            self._ok = True
            return
        # Never quietly answer with a model the creator did not ask for.
        self._error = (f"Ollama is running but {self.model} is not pulled "
                       f"(run: ollama pull {self.model})")

    def status(self) -> str:
        return f"ready ({self.model}, local, free)" if self._ok else self._error

    # -- plumbing ----------------------------------------------------------
    def _ask(self, system: str, prompt: str, task: tasks.TaskSpec) -> str:
        if not self._ok:
            raise RuntimeError(self._error or "Ollama unavailable")
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "num_predict": task.max_tokens},
        }
        if task.wants_json:
            payload["format"] = "json"       # small models need the constraint
        # Signal two of the routing judge.  Ollama returns a mean token
        # log-probability when asked; a runtime that does not is treated as
        # "cannot say" rather than "bad", so asking costs nothing where it is
        # not supported.
        payload["logprobs"] = True
        # Qwen3 and its relatives think before answering, and the thinking
        # comes out of the same token budget as the answer.  Asked for a
        # short structured reply with thinking left on, the whole budget goes
        # on deliberation and ``content`` arrives empty.
        payload["think"] = False
        request = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.last_seconds = time.monotonic() - started
        self.last_mean_logprob = _mean_logprob(data.get("logprobs"))
        return str(data.get("message", {}).get("content", "")).strip()

class LlamaCppLLM(SmallModelLLM):
    """A GGUF model loaded in process by ``llama-cpp-python``.

    The model is loaded on first use, not at startup: a few gigabytes of
    weights should not be paged in to answer a probe.
    """

    def __init__(self, gguf: str = "", strength: int = 40,
                 context: int = 4096, threads: int = 0) -> None:
        self.strength = strength
        self.context = context
        self.threads = threads
        self._llama = None
        self._error = ""
        self._ok = False
        self.path: Optional[Path] = None
        self.name = "llamacpp"
        try:
            import llama_cpp                                     # noqa: F401
        except Exception as exc:                                 # noqa: BLE001
            self._error = (f"llama-cpp-python not installed "
                           f"({exc.__class__.__name__})")
            return
        candidate = Path(gguf) if gguf else None
        if candidate is None or not candidate.exists():
            found = sorted(models_dir().glob("*.gguf"))
            candidate = found[0] if found else None
        if candidate is None:
            self._error = f"no .gguf model in {models_dir()}"
            return
        self.path = candidate
        self.name = f"llamacpp:{candidate.stem}"
        self._ok = True

    def status(self) -> str:
        if not self._ok:
            return self._error
        loaded = "loaded" if self._llama is not None else "not yet loaded"
        return f"ready ({self.path.name}, local, free, {loaded})"

    def _model(self):
        if self._llama is None:
            from llama_cpp import Llama                          # type: ignore
            log.info("loading %s", self.path)
            kwargs: Dict[str, Any] = {"model_path": str(self.path),
                                      "n_ctx": self.context, "verbose": False}
            if self.threads:
                kwargs["n_threads"] = self.threads
            self._llama = Llama(**kwargs)
        return self._llama

    def _ask(self, system: str, prompt: str, task: tasks.TaskSpec) -> str:
        if not self._ok:
            raise RuntimeError(self._error or "llama.cpp unavailable")
        kwargs: Dict[str, Any] = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "max_tokens": task.max_tokens,
            "temperature": 0.2,
        }
        if task.wants_json:
            kwargs["response_format"] = {"type": "json_object"}
        result = self._model().create_chat_completion(**kwargs)
        choices = result.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()
