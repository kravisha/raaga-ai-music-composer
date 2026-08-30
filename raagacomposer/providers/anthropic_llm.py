"""Anthropic language-model adapter (optional).

Enabled when the ``anthropic`` package is installed and a key is available
from the environment or the credentials file.  Nothing is hard-coded and the
application works fully without it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from .base import LLMProvider

log = get_logger("providers.anthropic")

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicLLM(LLMProvider):
    name = "anthropic"
    requires_key = True

    def __init__(self, model: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None) -> None:
        self.model = model
        self._client = None
        self._error = ""
        key = api_key or Settings.secret("anthropic_api_key")
        if not key:
            self._error = "No API key configured."
            return
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:  # noqa: BLE001
            self._error = f"anthropic package unavailable: {exc}"
            log.info(self._error)

    @property
    def available(self) -> bool:
        return self._client is not None

    def status(self) -> str:
        return f"ready ({self.model})" if self.available else self._error

    # -- plumbing ----------------------------------------------------------
    def _ask(self, system: str, prompt: str, max_tokens: int = 1200) -> str:
        if self._client is None:
            raise RuntimeError(self._error or "Anthropic provider unavailable")
        response = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}])
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _json(text: str) -> Any:
        match = re.search(r"[\[{].*[\]}]", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return None

    # -- capabilities ------------------------------------------------------
    def write_lyrics(self, slots: Sequence[Any], brief: Any) -> List[str]:
        lines = []
        for i, slot in enumerate(slots, start=1):
            pattern = "".join("X" if s else "." for s in slot.stresses)
            lines.append(f"{i}. section={slot.section_name} "
                         f"syllables={slot.syllable_count} stress={pattern}")
        system = (
            "You write song lyrics that fit an existing melody exactly. "
            "Each line must have precisely the requested number of syllables. "
            "Stressed positions (X) must fall on naturally stressed syllables. "
            "Write in the requested language, transliterated into Roman script "
            "so it can be sung by a synthesiser. Return JSON only: a list of "
            "strings, one per numbered line, in order.")
        prompt = (
            f"Language: {brief.language}\n"
            f"Mood: {brief.mood}\nFeel: {brief.feel}\n"
            f"Situation: {brief.situation}\nNotes: {brief.notes}\n\n"
            f"Lines to write:\n" + "\n".join(lines))
        data = self._json(self._ask(system, prompt, max_tokens=2000))
        if isinstance(data, list):
            return [str(x) for x in data]
        return []

    def classify_intent(self, text: str, intents: Sequence[str]) -> Dict[str, Any]:
        system = ("You classify a music director's spoken instruction into one "
                  "intent from a fixed list. Return JSON only: "
                  '{"intent": "...", "confidence": 0.0-1.0, "instrument": "..."}. '
                  "Use \"unknown\" if nothing fits.")
        prompt = f"Instruction: {text!r}\nAllowed intents: {', '.join(intents)}"
        data = self._json(self._ask(system, prompt, max_tokens=300))
        return data if isinstance(data, dict) else {}

    def suggest_raagas(self, brief: Any, candidates: Sequence[str]
                      ) -> List[Dict[str, str]]:
        system = ("You are a Carnatic and Hindustani music adviser. Choose raagas "
                  "from the supplied list only. Return JSON only: a list of "
                  '{"raaga": "...", "reason": "one sentence"}, best first.')
        prompt = (f"Brief: mood={brief.mood}; feel={brief.feel}; "
                  f"situation={brief.situation}; language={brief.language}\n"
                  f"Available raagas: {', '.join(candidates)}")
        data = self._json(self._ask(system, prompt, max_tokens=800))
        return data if isinstance(data, list) else []

    def suggest_instruments(self, description: str,
                            catalog: Sequence[str]) -> List[str]:
        system = ("Choose instruments for a described feel. Pick only from the "
                  "supplied list. Return JSON only: a list of instrument keys, "
                  "best first, at most four.")
        prompt = f"Feel: {description}\nAvailable: {', '.join(catalog)}"
        data = self._json(self._ask(system, prompt, max_tokens=300))
        if isinstance(data, list):
            return [str(x) for x in data if str(x) in set(catalog)]
        return []

    def explain(self, question: str, context: str = "") -> str:
        system = ("You are the arranger sitting beside a music director. Answer "
                  "in at most three sentences, practically.")
        return self._ask(system, f"{context}\n\n{question}" if context else question,
                         max_tokens=400)
