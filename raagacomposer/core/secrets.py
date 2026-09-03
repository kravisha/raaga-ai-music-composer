"""Secret storage (spec sections 42, 55).

Resolution order for reading any secret, in this order:

1. environment variable (read-only from here -- ``setx`` keeps working)
2. the OS keyring (Windows Credential Manager on this machine)
3. ``credentials.json`` in the application config directory
4. absent -- the caller reports itself unavailable

Writing always lands in exactly one of keyring or file, never both: ``set``
removes the name from the file the moment it lands in the keyring, and
``delete`` clears both, so a stale copy in the loser can never resurface as
the resolved value later.

``RAAGA_SECRET_BACKEND=file`` forces the file backend for everything below
the environment.  It is set unconditionally in ``tests/conftest.py`` --
before this module or ``keyring`` is ever imported -- so the automated suite
can never write to whichever developer's machine happens to run it.  A test
that needs to exercise the keyring branch passes an explicit
``keyring_backend=`` object to :class:`SecretStore` instead: that bypasses
the environment check entirely, because a test double can never touch a real
credential store, so there is nothing for the safety switch to guard against.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .logging_setup import get_logger

log = get_logger("secrets")

SERVICE_NAME = "RaagaComposer"


def _env_var(name: str) -> str:
    from .settings import ENV_KEYS
    return ENV_KEYS.get(name, name.upper())


def credentials_path():
    from .settings import config_dir
    return config_dir() / "credentials.json"


def _read_file() -> Dict[str, Any]:
    import json

    p = credentials_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_file(data: Dict[str, Any]) -> None:
    import json

    p = credentials_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass


class SecretStore:
    """One name, one value, resolved through environment / keyring / file."""

    def __init__(self, keyring_backend: Any = None) -> None:
        # An explicit backend (real or fake) is used exactly as given and
        # skips the RAAGA_SECRET_BACKEND gate below -- that gate exists to
        # keep *this process's own* keyring lookup off the real Credential
        # Manager during tests; it has nothing to protect against once the
        # caller has already handed in the object to use.
        self._forced_backend = keyring_backend
        self._warned = False

    # -- keyring resolution --------------------------------------------
    def _warn_fallback_once(self, reason: str) -> None:
        if not self._warned:
            log.info("secret storage: using credentials.json (%s)", reason)
            self._warned = True

    def _keyring_module(self) -> Optional[Any]:
        if self._forced_backend is not None:
            return self._forced_backend
        if os.environ.get("RAAGA_SECRET_BACKEND", "").lower() == "file":
            return None
        try:
            import keyring
        except Exception as exc:                                # noqa: BLE001
            self._warn_fallback_once(f"keyring not installed ({exc.__class__.__name__})")
            return None
        try:
            backend = keyring.get_keyring()
            backend_kind = type(backend).__name__.lower()
        except Exception as exc:                                 # noqa: BLE001
            self._warn_fallback_once(f"keyring backend probe failed: {exc}")
            return None
        if "fail" in backend_kind or backend_kind == "keyring" or "null" in backend_kind:
            # keyring.backends.fail.Keyring and .null.Keyring both name their
            # class "Keyring" (or subclass it); either means nothing usable
            # is actually configured on this machine.
            self._warn_fallback_once(f"no usable keyring backend ({backend_kind})")
            return None
        return keyring

    def backend_available(self) -> Dict[str, bool]:
        return {"keyring": self._keyring_module() is not None}

    def where_writes_go(self) -> str:
        return "keyring" if self._keyring_module() is not None else "file"

    # -- reads -----------------------------------------------------------
    def get(self, name: str) -> str:
        env = os.environ.get(_env_var(name), "")
        if env:
            return env.strip()
        kr = self._keyring_module()
        if kr is not None:
            try:
                value = kr.get_password(SERVICE_NAME, name)
            except Exception as exc:                             # noqa: BLE001
                self._warn_fallback_once(f"keyring read failed: {exc}")
                value = None
            if value:
                return str(value).strip()
        return str(_read_file().get(name, "")).strip()

    def source(self, name: str) -> str:
        if os.environ.get(_env_var(name), ""):
            return "environment"
        kr = self._keyring_module()
        if kr is not None:
            try:
                if kr.get_password(SERVICE_NAME, name):
                    return "keyring"
            except Exception as exc:                             # noqa: BLE001
                self._warn_fallback_once(f"keyring read failed: {exc}")
        if _read_file().get(name):
            return "file"
        return "none"

    # -- writes ------------------------------------------------------------
    def set(self, name: str, value: str) -> None:
        if not value:
            self.delete(name)
            return
        kr = self._keyring_module()
        if kr is not None:
            try:
                kr.set_password(SERVICE_NAME, name, value)
            except Exception as exc:                             # noqa: BLE001
                self._warn_fallback_once(f"keyring write failed: {exc}")
            else:
                data = _read_file()
                if name in data:
                    del data[name]
                    _write_file(data)
                return
        data = _read_file()
        data[name] = value
        _write_file(data)

    def delete(self, name: str) -> None:
        kr = self._keyring_module()
        if kr is not None:
            try:
                kr.delete_password(SERVICE_NAME, name)
            except Exception:                                    # noqa: BLE001
                pass  # not stored there is the routine case
        data = _read_file()
        if name in data:
            del data[name]
            _write_file(data)
