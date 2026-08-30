"""Local providers: the always-available engines.

These wrap the built-in synthesis and singing code so that the application has
a complete, working path with no credentials, no network and no downloads.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..core.logging_setup import get_logger
from ..music import instruments as catalog
from ..music.synth import render_notes
from ..voice import renderer as voice_renderer
from .base import LLMProvider, MusicProvider, VoiceProvider

log = get_logger("providers.local")


class LocalMusicProvider(MusicProvider):
    name = "local-synth"

    def render_part(self, notes: Sequence[Any], instrument: str, sample_rate: int,
                    total_seconds: Optional[float] = None, gain: float = 1.0,
                    seed: int = 0) -> np.ndarray:
        inst = catalog.get(instrument)
        if inst is None:
            raise KeyError(f"Unknown instrument: {instrument}")
        return render_notes(notes, inst, sample_rate, total_seconds=total_seconds,
                            gain=gain, seed=seed)

    def instruments(self) -> List[str]:
        return catalog.keys()

    def status(self) -> str:
        return f"{len(catalog.keys())} built-in instruments"


class LocalVoiceProvider(VoiceProvider):
    name = "local-voice"

    def render_vocal(self, melody: Any, lyrics: Any, profile: Any, direction: Any,
                     sample_rate: int, total_seconds: Optional[float] = None,
                     seed: int = 0) -> np.ndarray:
        return voice_renderer.render_melody(melody, lyrics, profile, direction,
                                            sample_rate, total_seconds, seed)

    def voices(self) -> List[str]:
        from ..voice.profiles import BUILTIN
        return [v.name for v in BUILTIN]

    def status(self) -> str:
        return "formant singing synthesiser"


class LocalLLM(LLMProvider):
    """No language model configured: the rule and lexicon engines are used."""

    name = "local-rules"

    @property
    def available(self) -> bool:
        return False

    def status(self) -> str:
        return "not configured - rule and lexicon engines are in use"

    def write_lyrics(self, slots, brief) -> List[str]:
        return []

    def classify_intent(self, text: str, intents) -> Dict[str, Any]:
        return {}

    def suggest_raagas(self, brief, candidates) -> List[Dict[str, str]]:
        return []

    def suggest_instruments(self, description: str, catalog_keys) -> List[str]:
        return []

    def explain(self, question: str, context: str = "") -> str:
        return ""
