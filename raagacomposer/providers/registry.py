"""Provider registry (spec section 11).

Builds the provider set from settings, always with a working local fallback.
"""
from __future__ import annotations

from typing import Optional

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from .base import LLMProvider, MusicProvider, ProviderSet, VoiceProvider
from .local import LocalLLM, LocalMusicProvider, LocalVoiceProvider

log = get_logger("providers")


def build_llm(settings: Settings) -> LLMProvider:
    choice = (settings.llm_provider or "auto").lower()
    if choice in ("auto", "anthropic"):
        try:
            from .anthropic_llm import AnthropicLLM
            provider = AnthropicLLM()
            if provider.available:
                log.info("LLM provider: %s", provider.status())
                return provider
            if choice == "anthropic":
                log.warning("anthropic requested but unavailable: %s",
                            provider.status())
        except Exception as exc:  # noqa: BLE001
            log.warning("anthropic adapter failed to load: %s", exc)
    return LocalLLM()


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
            "engines. Set ANTHROPIC_API_KEY to enable richer lyric writing.")
    return providers
