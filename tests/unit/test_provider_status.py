"""Unit tests: the provider status model (spec section 41, TEST H)."""
from __future__ import annotations

import pytest

from raagacomposer.core.settings import Settings
from raagacomposer.providers import registry
from raagacomposer.providers.status import provider_statuses

pytestmark = pytest.mark.unit


def _no_key(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "secret", classmethod(lambda cls, n: ""))


def _fake_key(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "secret",
                        classmethod(lambda cls, n: "sk-ant-not-a-real-key"))


def test_no_key_claude_is_not_configured(settings, monkeypatch):
    _no_key(monkeypatch)
    settings.llm_provider = "claude"
    settings.llm_routing = "auto"      # the fixture default ("off") would
                                       # report "Off" instead - see the
                                       # dedicated routing test below
    llm = registry.build_llm(settings)
    rows = provider_statuses(llm, settings)
    claude_rows = [r for r in rows if r.kind == "cloud"]
    assert claude_rows
    assert all(r.state == "Not configured" for r in claude_rows)


def test_a_key_makes_claude_configured_with_no_network_call(settings, monkeypatch):
    _fake_key(monkeypatch)
    settings.llm_provider = "claude"
    settings.llm_routing = "auto"

    calls = []
    import urllib.request
    real_urlopen = urllib.request.urlopen

    def guarded(*a, **k):
        calls.append((a, k))
        raise AssertionError("provider_statuses must not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
    try:
        llm = registry.build_llm(settings)
        rows = provider_statuses(llm, settings)
    finally:
        monkeypatch.setattr(urllib.request, "urlopen", real_urlopen)

    assert not calls
    claude_rows = [r for r in rows if r.kind == "cloud"]
    assert claude_rows
    assert all(r.state == "Configured" for r in claude_rows)
    assert all(r.model for r in claude_rows)


def test_routing_off_reports_off_regardless_of_configuration(settings, monkeypatch):
    _fake_key(monkeypatch)
    settings.llm_provider = "claude"
    settings.llm_routing = "off"
    llm = registry.build_llm(settings)
    rows = provider_statuses(llm, settings)
    # llm_provider="claude" builds no local LLM backend at all, only the two
    # Claude tiers - the speech rows are also kind="local" but are outside
    # llm_routing's authority, so they are excluded here on purpose.
    claude_rows = [r for r in rows if r.kind == "cloud"]
    assert claude_rows
    assert all(r.state == "Off" for r in claude_rows)


def test_builtin_engines_are_always_ready(settings, monkeypatch):
    _no_key(monkeypatch)
    settings.llm_provider = "off"
    llm = registry.build_llm(settings)
    rows = provider_statuses(llm, settings)
    builtin = [r for r in rows if r.kind == "builtin"]
    assert builtin
    assert all(r.state == "Ready" for r in builtin)


def test_local_backend_not_installed_when_nothing_is_running(settings, monkeypatch):
    _no_key(monkeypatch)
    settings.llm_provider = "ollama"
    settings.llm_routing = "auto"
    settings.llm_local_endpoint = "http://127.0.0.1:1"      # refused immediately
    llm = registry.build_llm(settings)
    rows = provider_statuses(llm, settings)
    local_rows = [r for r in rows if r.kind == "local" and r.name.startswith("ollama")]
    assert local_rows
    assert local_rows[0].state == "Not installed"


def test_accepts_a_provider_set_directly(settings, monkeypatch):
    _no_key(monkeypatch)
    settings.llm_provider = "off"
    providers = registry.build(settings, stt_name="typed")
    rows = provider_statuses(providers, settings)
    assert any(r.kind == "builtin" for r in rows)


def test_speech_backends_are_listed(settings, monkeypatch):
    _no_key(monkeypatch)
    settings.llm_provider = "off"
    llm = registry.build_llm(settings)
    rows = provider_statuses(llm, settings)
    names = {r.name for r in rows}
    assert "vosk" in names and "whisper" in names
