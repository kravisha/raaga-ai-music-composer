"""Integration: the Settings dialog, built offscreen against a real
AppController (spec sections 41, 42, TEST H).

Never touches the network: Validate is exercised with the ``anthropic``
client monkeypatched, and every other case stays local (file secret backend,
forced by tests/conftest.py's RAAGA_SECRET_BACKEND=file).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication            # noqa: E402

from raagacomposer.core.secrets import SecretStore     # noqa: E402
from raagacomposer.providers.status import ANTHROPIC_KEY_NAME  # noqa: E402
from raagacomposer.ui.settings_dialog import SettingsDialog     # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clean_secret():
    """The ``app``/``settings`` fixtures share one config directory across
    the whole session (see tests/conftest.py), so a key saved by one test
    would otherwise still be there for the next one."""
    SecretStore().delete(ANTHROPIC_KEY_NAME)
    try:
        yield
    finally:
        SecretStore().delete(ANTHROPIC_KEY_NAME)


@pytest.fixture
def dialog(qt_app, app):
    dlg = SettingsDialog(app)
    try:
        yield dlg
    finally:
        dlg.close()


def test_dialog_builds_with_no_key(dialog):
    assert dialog.windowTitle() == "Settings"
    assert "Not configured" in dialog.storage_label.text()
    assert dialog.table.rowCount() > 0
    # builtin engines are always present and Ready, with or without a key
    kinds = [dialog.table.item(row, 1).text() for row in range(dialog.table.rowCount())]
    assert "builtin" in kinds


def test_save_stores_through_the_file_backend_and_updates_the_label(dialog, app):
    dialog.key_field.setText("sk-ant-fake0123456789fake")
    dialog._on_save()
    assert SecretStore().get(ANTHROPIC_KEY_NAME) == "sk-ant-fake0123456789fake"
    assert SecretStore().source(ANTHROPIC_KEY_NAME) == "file"
    assert "credentials.json" in dialog.storage_label.text()
    assert dialog.key_field.text() == ""                # cleared, never re-shown


def test_save_rejects_a_key_with_the_wrong_shape(dialog, monkeypatch):
    warned = []
    import raagacomposer.ui.settings_dialog as mod
    monkeypatch.setattr(mod.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    dialog.key_field.setText("not-a-real-key")
    dialog._on_save()
    assert warned
    assert SecretStore().get(ANTHROPIC_KEY_NAME) == ""


def test_remove_clears_the_stored_key(dialog, monkeypatch):
    SecretStore().set(ANTHROPIC_KEY_NAME, "sk-ant-fake0123456789fake")
    import raagacomposer.ui.settings_dialog as mod
    monkeypatch.setattr(mod.QMessageBox, "question",
                        lambda *a, **k: mod.QMessageBox.Yes)
    dialog._on_remove()
    assert SecretStore().get(ANTHROPIC_KEY_NAME) == ""
    assert "Not configured" in dialog.storage_label.text()


def test_validate_reports_failure_without_leaking_the_key(dialog, monkeypatch, qt_app):
    class FakeAuthError(Exception):
        pass

    class FakeModels:
        def list(self, limit=1):
            raise FakeAuthError("invalid x-api-key")

    class FakeClient:
        def __init__(self, api_key=""):
            self.api_key = api_key
            self.models = FakeModels()

    class FakeAnthropicModule:
        Anthropic = FakeClient

    import sys
    monkeypatch.setitem(sys.modules, "anthropic", FakeAnthropicModule())

    dialog.key_field.setText("sk-ant-fake0123456789fake")
    dialog._on_validate()
    worker = dialog._worker
    assert worker is not None
    worker.wait(5000)
    qt_app.processEvents()

    text = dialog.validate_label.text()
    assert "sk-ant-fake0123456789fake" not in text
    assert "FakeAuthError" in text or "invalid x-api-key" in text
