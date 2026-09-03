"""Unit tests: the secret store (spec sections 42, 55).

``tests/conftest.py`` sets ``RAAGA_SECRET_BACKEND=file`` for the whole suite,
so a bare ``SecretStore()`` here always exercises the credentials.json path.
The keyring branch is tested separately with a fake backend object injected
directly into the store -- never the real one.
"""
from __future__ import annotations

import pytest

from raagacomposer.core.secrets import SecretStore

pytestmark = pytest.mark.unit

NAME = "anthropic_api_key"


class FakeKeyring:
    """A dict-backed stand-in for the ``keyring`` module's public functions."""

    def __init__(self) -> None:
        self.store = {}

    def set_password(self, service, username, value):
        self.store[(service, username)] = value

    def get_password(self, service, username):
        return self.store.get((service, username))

    def delete_password(self, service, username):
        if (service, username) not in self.store:
            raise LookupError("not found")
        del self.store[(service, username)]


class RaisingKeyring(FakeKeyring):
    def set_password(self, service, username, value):
        raise RuntimeError("backend exploded")

    def get_password(self, service, username):
        raise RuntimeError("backend exploded")


# --------------------------------------------------------------------------
# file backend (forced by RAAGA_SECRET_BACKEND=file in conftest)
# --------------------------------------------------------------------------
def test_file_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    store = SecretStore()
    assert store.get(NAME) == ""
    assert store.source(NAME) == "none"

    store.set(NAME, "sk-ant-abc123")
    assert store.get(NAME) == "sk-ant-abc123"
    assert store.source(NAME) == "file"
    assert (tmp_path / "credentials.json").exists()

    store.delete(NAME)
    assert store.get(NAME) == ""
    assert store.source(NAME) == "none"


def test_environment_wins_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    store = SecretStore()
    store.set(NAME, "from-file")
    assert store.get(NAME) == "from-file"
    assert store.source(NAME) == "file"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert store.get(NAME) == "from-env"
    assert store.source(NAME) == "environment"


def test_where_writes_go_is_file_when_backend_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    store = SecretStore()
    assert store.backend_available() == {"keyring": False}
    assert store.where_writes_go() == "file"


# --------------------------------------------------------------------------
# keyring backend, always via an injected fake
# --------------------------------------------------------------------------
def test_set_prefers_keyring_and_removes_the_file_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    # A stale file copy from before keyring was available.
    file_store = SecretStore()
    file_store.set(NAME, "stale-file-value")
    assert (tmp_path / "credentials.json").read_text(encoding="utf-8").find(NAME) != -1

    fake = FakeKeyring()
    store = SecretStore(keyring_backend=fake)
    assert store.backend_available() == {"keyring": True}
    assert store.where_writes_go() == "keyring"

    store.set(NAME, "sk-ant-keyring-value")
    assert fake.store[("RaagaComposer", NAME)] == "sk-ant-keyring-value"
    # The file no longer holds this name: one copy of the secret exists.
    import json
    data = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    assert NAME not in data

    assert store.get(NAME) == "sk-ant-keyring-value"
    assert store.source(NAME) == "keyring"


def test_delete_clears_both_keyring_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    fake = FakeKeyring()
    store = SecretStore(keyring_backend=fake)
    store.set(NAME, "sk-ant-value")
    assert fake.store

    store.delete(NAME)
    assert (fake.store.get(("RaagaComposer", NAME)) is None)
    assert store.get(NAME) == ""
    assert store.source(NAME) == "none"


def test_keyring_raising_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    store = SecretStore(keyring_backend=RaisingKeyring())

    store.set(NAME, "sk-ant-fallback")
    assert store.get(NAME) == "sk-ant-fallback"
    assert store.source(NAME) == "file"

    import json
    data = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    assert data[NAME] == "sk-ant-fallback"
