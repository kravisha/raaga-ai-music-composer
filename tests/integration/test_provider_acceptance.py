"""TEST H (spec section 63): provider status with and without a key.

    1. Run without an Anthropic key.
    2. Confirm the app launches and local/core features work.
    3. Configure an Anthropic key.
    4. Validate key/provider status.
    5. (a complex reasoning task using Claude - exercised in
       tests/unit/test_provider_status.py and test_provider_routing.py,
       which confirm the router actually selects Claude once configured;
       this test's job is the controller-level status contract.)
    6. Remove/disable cloud access.
    7. Confirm compatible tasks can route locally (here: the builtin engines,
       since no local SLM runtime is installed in the test environment).

Isolated from every other test's shared config directory - a fresh
tmp_path-backed ``RAAGA_COMPOSER_HOME`` - because this is the one place in
the suite that flips ``llm_provider`` to "auto" and needs the Claude/local
backends to actually be built.
"""
from __future__ import annotations

import pytest

from raagacomposer.core.models import CreativeBrief
from raagacomposer.core.secrets import SecretStore
from raagacomposer.core.settings import Settings
from raagacomposer.providers.router import RoutedLLM
from raagacomposer.providers.status import ANTHROPIC_KEY_NAME
from raagacomposer.raaga.selection import suggest as suggest_raagas

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def provider_app(tmp_path, monkeypatch):
    """A real AppController with the router (not the "off" stub) built, in
    a config directory nobody else's test can see or leave state in."""
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RAAGA_SECRET_BACKEND", "file")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = Settings.load()
    settings.projects_dir = str(tmp_path / "projects")
    settings.knowledge_db = str(tmp_path / "knowledge.db")
    settings.knowledge_base_db = str(tmp_path / "knowledge_base.db")
    settings.stt_provider = "none"
    settings.llm_provider = "auto"
    settings.llm_routing = "auto"
    settings.llm_local_endpoint = "http://127.0.0.1:1"     # refused, no wait

    from raagacomposer.app import AppController
    controller = AppController(settings)
    try:
        yield controller
    finally:
        controller.close()


def _claude_states(app) -> set:
    return {r.state for r in app.provider_statuses() if r.kind == "cloud"}


def _builtin_states(app) -> set:
    return {r.state for r in app.provider_statuses() if r.kind == "builtin"}


def test_provider_status_with_and_without_a_key(provider_app):
    app = provider_app

    # 1-2. No key: the app has already launched (the fixture built it) and
    # is usable - Claude is not configured, the builtin engines are Ready.
    assert _claude_states(app) == {"Not configured"}
    assert _builtin_states(app) == {"Ready"}

    # 3-4. Configure the key and refresh - the router notices without a
    # restart, the same mechanism it uses to notice the key on its own
    # timer (docs/DECISIONS.md, "A key is noticed while running").
    SecretStore().set(ANTHROPIC_KEY_NAME, "sk-ant-not-a-real-key")
    assert isinstance(app.providers.llm, RoutedLLM)
    app.providers.llm.refresh()
    assert _claude_states(app) == {"Configured"}

    # 6. Remove/disable cloud access.
    SecretStore().delete(ANTHROPIC_KEY_NAME)
    app.providers.llm.refresh()
    assert _claude_states(app) == {"Not configured"}

    # 7. Compatible tasks still route - here, to the builtin raaga heuristic,
    # since no local SLM runtime is installed in this environment.  This is
    # the deterministic engine "suggest_raagas" falls back to whenever no
    # backend answers (providers/router.py's RoutedLLM._call).
    brief = CreativeBrief(situation="a man alone on a terrace after midnight",
                          mood="longing",
                          feel="lonely, late at night, but still warm",
                          language="Tamil", duration_target=90.0)
    suggestions = suggest_raagas(brief, app.raagas, limit=4)
    assert suggestions
    assert _builtin_states(app) == {"Ready"}
