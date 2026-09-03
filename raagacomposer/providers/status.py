"""Provider status model (spec section 41).

Turns what the registry already knows -- a :class:`RoutedLLM` and the probes
its backends ran at construction time -- into the vocabulary section 41 asks
for: ``Configured`` / ``Not configured`` / ``Unavailable`` for a cloud
provider, ``Available`` / ``Not installed`` / ``Loading`` / ``Ready`` for a
local one, and ``Off`` for anything the current routing policy has excluded.
Nothing here makes a network call beyond the ones the registry already made
when it built the providers (Ollama's probe is a local HTTP call; nothing
else reaches out).

The builtin rule and lexicon engines have no "provider" behind them at all --
they are plain Python -- so they are always ``Ready``: they need nothing
configured, installed or reachable to answer.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from ..core.secrets import SecretStore
from ..core.settings import Settings
from .router import RoutedLLM

ANTHROPIC_KEY_NAME = "anthropic_api_key"


@dataclass
class ProviderStatus:
    name: str
    kind: str            # "cloud" | "local" | "builtin"
    state: str            # see module docstring for the vocabulary per kind
    detail: str = ""
    model: str = ""
    source: str = "none"  # where a cloud credential came from, else "none"


def _policy_allows(policy: str, is_local: bool) -> bool:
    """Whether the current routing policy would ever hand this backend work.

    Mirrors :meth:`RoutedLLM._rank_all`'s policy branch, kept separate because
    the status table wants to say *why* a perfectly-working backend is not in
    use, which the router itself has no reason to distinguish.
    """
    if policy == "off":
        return False
    if policy == "local_only":
        return is_local
    if policy == "claude_only":
        return not is_local
    return True


def _backend_model(backend: Any) -> str:
    model = getattr(backend, "model", "")
    if model:
        return str(model)
    path = getattr(backend, "path", None)
    if path:
        return Path(path).stem
    return ""


def _cloud_row(backend: Any, policy: str, key_present: bool,
              source: str) -> ProviderStatus:
    if not _policy_allows(policy, is_local=False):
        state = "Off"
    elif backend.available:
        state = "Configured"
    elif not key_present:
        state = "Not configured"
    else:
        state = "Unavailable"
    return ProviderStatus(name=backend.name, kind="cloud", state=state,
                          detail=backend.status(), model=_backend_model(backend),
                          source=source)


def _local_row(backend: Any, policy: str) -> ProviderStatus:
    if not _policy_allows(policy, is_local=True):
        state = "Off"
    elif backend.available:
        state = "Ready"
    else:
        state = "Not installed"
    return ProviderStatus(name=backend.name, kind="local", state=state,
                          detail=backend.status(), model=_backend_model(backend),
                          source="none")


def _resolve_llm(registry_or_router: Any) -> Optional[Any]:
    """Accept either a :class:`ProviderSet`-like object or a bare LLM."""
    llm = getattr(registry_or_router, "llm", registry_or_router)
    return llm


def _llm_rows(registry_or_router: Any, settings: Settings,
             key_present: bool, source: str) -> List[ProviderStatus]:
    llm = _resolve_llm(registry_or_router)
    if isinstance(llm, RoutedLLM):
        policy = llm.policy
        rows: List[ProviderStatus] = []
        for backend in llm.backends:
            if getattr(backend, "is_local", False):
                rows.append(_local_row(backend, policy))
            else:
                rows.append(_cloud_row(backend, policy, key_present, source))
        return rows
    # ``llm_provider`` (not the routing policy) turned the whole language
    # model subsystem off, or nothing could be built at all: one row says so
    # rather than pretending a provider exists to report on.
    detail = llm.status() if llm is not None else "no language model built"
    return [ProviderStatus(name="claude", kind="cloud", state="Off",
                           detail=detail, model="", source=source)]


def _speech_rows(stt_adapter: Optional[Any]) -> List[ProviderStatus]:
    active_name = getattr(stt_adapter, "name", "") if stt_adapter else ""
    active_available = bool(stt_adapter and stt_adapter.available)
    active_detail = stt_adapter.status() if stt_adapter else ""

    rows = []
    for name, module in (("vosk", "vosk"), ("whisper", "faster_whisper")):
        if active_name == name and active_available:
            rows.append(ProviderStatus(name=name, kind="local", state="Ready",
                                       detail=active_detail, source="none"))
            continue
        installed = importlib.util.find_spec(module) is not None
        if installed:
            rows.append(ProviderStatus(
                name=name, kind="local", state="Available",
                detail="installed, not the active speech backend"
                       if active_name and active_name != name
                       else "installed; no model probed yet",
                source="none"))
        else:
            rows.append(ProviderStatus(name=name, kind="local",
                                       state="Not installed",
                                       detail=f"{module} package not installed",
                                       source="none"))
    return rows


def _builtin_rows() -> List[ProviderStatus]:
    return [
        ProviderStatus(name="rule-engine", kind="builtin", state="Ready",
                      detail="deterministic intent rules - always available",
                      source="none"),
        ProviderStatus(name="lexicon-engine", kind="builtin", state="Ready",
                      detail="deterministic lyric fitting and raga heuristics",
                      source="none"),
    ]


def provider_statuses(registry_or_router: Any, settings: Settings,
                      stt_adapter: Optional[Any] = None
                      ) -> List[ProviderStatus]:
    """The full status table: language models, speech, and the builtin engines.

    ``registry_or_router`` is whatever the controller already holds -- a
    :class:`~raagacomposer.providers.base.ProviderSet` (read via its ``.llm``)
    or a bare :class:`RoutedLLM` / :class:`LLMProvider` -- so the caller does
    not need to unwrap it first.
    """
    key_present = bool(Settings.secret(ANTHROPIC_KEY_NAME))
    source = SecretStore().source(ANTHROPIC_KEY_NAME)
    rows = _llm_rows(registry_or_router, settings, key_present, source)
    rows.extend(_speech_rows(stt_adapter))
    rows.extend(_builtin_rows())
    return rows
