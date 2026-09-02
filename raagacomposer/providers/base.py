"""Provider abstraction layer (spec section 11).

Every external service sits behind one of these interfaces.  Provider-specific
code exists only in the adapter modules; the application talks to these types
and never to an SDK directly, so a provider can be swapped in settings without
touching the workflow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class ProviderInfo:
    name: str
    kind: str                    # llm | music | voice | stt
    available: bool
    detail: str = ""
    requires_key: bool = False


class Provider:
    name = "provider"
    kind = "generic"
    requires_key = False

    @property
    def available(self) -> bool:
        return True

    def info(self) -> ProviderInfo:
        return ProviderInfo(self.name, self.kind, self.available,
                            self.status(), self.requires_key)

    def status(self) -> str:
        return "ready" if self.available else "unavailable"


class LLMProvider(Provider):
    """Text intelligence: lyrics, raaga reasoning, fuzzy intent.

    The three attributes below are what the router sorts on.  They are
    deliberately coarse: their job is to order a handful of backends, not to
    predict anything.  ``strength`` is a rough capability score out of 100 and
    ``cost_per_mtok`` is a blended USD price per million tokens, assuming three
    input tokens per output token - the shape of every request this
    application makes.  A local model costs nothing to run, so it scores 0 and
    sorts first whenever price is the tie-breaker.
    """

    kind = "llm"
    is_local = False             # runs on this machine: no network, no bill
    strength = 50                # rough capability out of 100, for ordering
    cost_per_mtok = 0.0          # blended USD per million tokens; 0 = free

    def write_lyrics(self, slots: Sequence[Any], brief: Any) -> List[str]:
        raise NotImplementedError

    def classify_intent(self, text: str, intents: Sequence[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def suggest_raagas(self, brief: Any, candidates: Sequence[str]) -> List[Dict[str, str]]:
        raise NotImplementedError

    def suggest_instruments(self, description: str,
                            catalog: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def explain(self, question: str, context: str = "") -> str:
        raise NotImplementedError


class MusicProvider(Provider):
    """Instrument rendering for a set of notes."""

    kind = "music"

    def render_part(self, notes: Sequence[Any], instrument: str, sample_rate: int,
                    total_seconds: Optional[float] = None,
                    gain: float = 1.0, seed: int = 0) -> np.ndarray:
        raise NotImplementedError

    def instruments(self) -> List[str]:
        raise NotImplementedError


class VoiceProvider(Provider):
    """Singing synthesis / authorised voice conversion."""

    kind = "voice"

    def render_vocal(self, melody: Any, lyrics: Any, profile: Any, direction: Any,
                     sample_rate: int, total_seconds: Optional[float] = None,
                     seed: int = 0) -> np.ndarray:
        raise NotImplementedError

    def voices(self) -> List[str]:
        raise NotImplementedError


@dataclass
class ProviderSet:
    llm: Optional[LLMProvider] = None
    music: Optional[MusicProvider] = None
    voice: Optional[VoiceProvider] = None
    stt_name: str = ""
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        rows = []
        for p in (self.llm, self.music, self.voice):
            if p is None:
                continue
            info = p.info()
            rows.append(f"{info.kind:6s} {info.name:<16} "
                        f"{'ready' if info.available else 'unavailable'}"
                        f"  {info.detail}")
        if self.stt_name:
            rows.append(f"stt    {self.stt_name}")
        # When several language models are configured, which one answers what
        # is the first thing anyone asks - so it belongs in the same place the
        # providers themselves are reported.
        routing = getattr(self.llm, "explain_routing", None)
        if callable(routing) and self.llm is not None and self.llm.available:
            rows.append("")
            rows.append("routing (task, complexity, backends in order tried):")
            rows.extend("  " + line for line in routing().splitlines())
        return "\n".join(rows)
