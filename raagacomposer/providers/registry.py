"""Provider registry (spec section 11).

Builds the provider set from settings, always with a working local fallback.

The language-model side is assembled here as a list of *factories* rather than
instances, and handed to the router.  That indirection is what lets the router
rebuild a backend later - when a key appears, or the creator starts Ollama -
without the application being restarted or a line being changed.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from ..core.logging_setup import get_logger
from ..core.settings import Settings
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


def _local_factories(settings: Settings) -> List[Callable[[], LLMProvider]]:
    from .local_llm import LlamaCppLLM, OllamaLLM

    return [
        lambda: OllamaLLM(endpoint=settings.llm_local_endpoint,
                          model=settings.llm_local_model,
                          strength=settings.llm_local_strength,
                          timeout=settings.llm_local_timeout),
        lambda: LlamaCppLLM(gguf=settings.llm_local_gguf,
                            strength=settings.llm_local_strength),
    ]


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
    if choice in ("auto", "ollama", "llamacpp"):
        local = _local_factories(settings)
        if choice == "ollama":
            local = local[:1]
        elif choice == "llamacpp":
            local = local[1:]
        factories += local

    router = RoutedLLM(factories, policy=policy,
                       refresh_seconds=settings.llm_refresh_seconds)
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
