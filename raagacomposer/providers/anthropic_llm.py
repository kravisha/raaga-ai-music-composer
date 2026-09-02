"""Deprecated module name, kept so older settings and imports keep working.

The adapter moved to :mod:`raagacomposer.providers.claude_llm` when it gained
model tiering and a router above it.  ``AnthropicLLM`` remains as an alias for
:class:`~raagacomposer.providers.claude_llm.ClaudeLLM`; new code should import
the new name.
"""
from __future__ import annotations

from .claude_llm import DEFAULT_MODEL, MODELS, ClaudeLLM

AnthropicLLM = ClaudeLLM

__all__ = ["AnthropicLLM", "ClaudeLLM", "DEFAULT_MODEL", "MODELS"]
