"""Provider registry (spec section 11).

Builds the provider set from settings, always with a working local fallback.

The language-model side is assembled here as a list of *factories* rather than
instances, and handed to the router.  That indirection is what lets the router
rebuild a backend later - when a key appears, or the creator starts Ollama -
without the application being restarted or a line being changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from ..core.logging_setup import get_logger
from ..core.settings import Settings, config_dir
from . import escalation
from .base import LLMProvider, MusicProvider, ProviderSet, VoiceProvider
from .local import LocalLLM, LocalMusicProvider, LocalVoiceProvider
from .router import RoutedLLM

log = get_logger("providers")

#: Older settings files, and the spec, call these something else.
ALIASES = {"local": "off", "anthropic": "claude", "none": "off"}


def _claude_factories(settings: Settings) -> List[Callable[[], LLMProvider]]:
    """The strong model and the cheap one, in that order.

    Two instances of one adapter.  The router picks between them per task;
    neither knows the other exists.
    """
    from .claude_llm import ClaudeLLM

    made: List[Callable[[], LLMProvider]] = []
    seen = set()
    for model in (settings.llm_claude_model, settings.llm_claude_light_model):
        if not model or model in seen:
            continue
        seen.add(model)
        made.append(lambda m=model: ClaudeLLM(
            model=m, effort=settings.llm_claude_effort,
            thinking=settings.llm_claude_thinking))
    return made


#: Roughly how capable each tier is, 0-100, for the ordering the router
#: already does.  Coarse on purpose, like every other strength here: the
#: job is to order a handful of candidates, not to predict anything.  What
#: these are actually worth is a question for the attempt-then-escalate
#: judge and the log it keeps, not for a number chosen in advance.
TIER_STRENGTH = {"small": 40, "json": 50, "mid": 55}


def _ollama_factories(settings: Settings) -> List[Callable[[], LLMProvider]]:
    """One backend per registered local tier, cheapest first.

    The tiers are named in ``settings.routing_tiers`` so the escalation loop
    can ask for the one it wants rather than guessing from a list.  Each is
    probed at construction and reports itself unavailable if it has not been
    pulled, so a tier named in the config but missing on the machine is a
    visible "not installed" rather than a silent substitution.
    """
    from .local_llm import OllamaLLM

    made: List[Callable[[], LLMProvider]] = []
    seen: set = set()
    tiers = settings.routing_tiers or {}
    # Cheapest first, then anything the config names that the orders do not.
    ordered = [t for t in (list(settings.routing_order or [])
                           + list(settings.routing_order_json or []))
               if t in tiers]
    ordered += [t for t in tiers if t not in ordered]
    for tier in ordered:
        model = tiers.get(tier, "")
        if not model or model in seen:
            continue
        seen.add(model)
        made.append(lambda m=model, t=tier: OllamaLLM(
            endpoint=settings.llm_local_endpoint, model=m,
            strength=TIER_STRENGTH.get(t, settings.llm_local_strength),
            timeout=settings.llm_local_timeout))

    # The single-model setting still works for anyone who set it, and for a
    # machine with no tiers configured at all.
    if settings.llm_local_model and settings.llm_local_model not in seen:
        made.append(lambda: OllamaLLM(endpoint=settings.llm_local_endpoint,
                                      model=settings.llm_local_model,
                                      strength=settings.llm_local_strength,
                                      timeout=settings.llm_local_timeout))
    return made


def _llamacpp_factories(settings: Settings) -> List[Callable[[], LLMProvider]]:
    from .local_llm import LlamaCppLLM

    return [lambda: LlamaCppLLM(gguf=settings.llm_local_gguf,
                                strength=settings.llm_local_strength)]


def _local_factories(settings: Settings) -> List[Callable[[], LLMProvider]]:
    return _ollama_factories(settings) + _llamacpp_factories(settings)


def build_llm(settings: Settings) -> LLMProvider:
    choice = (settings.llm_provider or "auto").lower()
    choice = ALIASES.get(choice, choice)
    policy = (settings.llm_routing or "auto").lower()

    factories: List[Callable[[], LLMProvider]] = []
    if choice == "off":
        # The built-in engines only.  Preserved as its own setting because
        # "do not call a model at all" is a legitimate choice, not a failure.
        return LocalLLM()
    if choice in ("auto", "claude"):
        factories += _claude_factories(settings)
    # Named rather than sliced.  This used to take local[:1] and local[1:],
    # which silently assumed the list was exactly [Ollama, LlamaCpp]; with a
    # backend per local tier that assumption gives "llamacpp" a set of Ollama
    # backends.
    if choice == "ollama":
        factories += _ollama_factories(settings)
    elif choice == "llamacpp":
        factories += _llamacpp_factories(settings)
    elif choice == "auto":
        factories += _local_factories(settings)

    log_path = settings.routing_log or str(config_dir() / "routing_attempts.jsonl")
    router = RoutedLLM(
        factories, policy=policy,
        refresh_seconds=settings.llm_refresh_seconds,
        thresholds=escalation.Thresholds.from_settings(settings),
        attempt_log=escalation.AttemptLog(Path(log_path)),
        tiers=dict(settings.routing_tiers or {}),
        order=list(settings.routing_order or []))
    if router.available:
        log.info("LLM routing: %s", router.status())
    return router


def build(settings: Optional[Settings] = None,
          stt_name: str = "") -> ProviderSet:
    settings = settings or Settings.load()
    providers = ProviderSet(
        llm=build_llm(settings),
        music=LocalMusicProvider(),
        voice=LocalVoiceProvider(),
        stt_name=stt_name,
    )
    if providers.llm is not None and not providers.llm.available:
        providers.notes.append(
            "No language model configured - lyrics and intent use the built-in "
            "engines. Set ANTHROPIC_API_KEY to enable Claude, or run a local "
            "model with Ollama.")
    return providers
