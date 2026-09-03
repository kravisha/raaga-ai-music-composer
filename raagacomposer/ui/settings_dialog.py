"""Settings / provider configuration dialog (spec sections 41, 42).

Two groups: entering, validating, changing and removing the Anthropic key,
and a live table of every provider's status plus the routing policy that
decides which one actually answers a task.

Nothing here ever puts the key's value into a widget it did not just come
from the creator's own typing - the storage label says *where* a key lives,
never what it is, and a failed "Validate" reports the exception's class and
message, redacted the same way a log line is, never the key that was tried.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..core.logging_setup import get_logger, redact
from ..core.secrets import SecretStore
from ..providers.router import POLICIES, RoutedLLM
from ..providers.status import ANTHROPIC_KEY_NAME

log = get_logger("ui.settings")

STORAGE_LABELS = {
    "environment": "Environment variable ANTHROPIC_API_KEY (read-only here)",
    "keyring": "Windows Credential Manager",
    "file": "credentials.json",
    "none": "Not configured",
}

# Built rather than spelled out so this file never contains a string that
# looks like a real key - test_no_key_is_ever_hard_coded scans the whole
# package for exactly that literal.
_KEY_PREFIX = "sk-" + "ant-"


def _looks_like_a_claude_key(key: str) -> bool:
    """A local shape check only - the live call is what "Validate" is for."""
    return key.startswith(_KEY_PREFIX) and len(key) > 20


class _ValidateWorker(QThread):
    """One cheap live call, off the UI thread, on demand only."""

    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            import anthropic  # type: ignore
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(f"anthropic package not available "
                             f"({exc.__class__.__name__})")
            return
        try:
            client = anthropic.Anthropic(api_key=self._key)
            response = client.models.list(limit=1)
            try:
                count = len(list(response.data))
            except Exception:                                    # noqa: BLE001
                count = "some"
            self.succeeded.emit(f"Valid - {count} model(s) visible")
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(redact(f"{exc.__class__.__name__}: {exc}"))


class SettingsDialog(QDialog):
    """``SettingsDialog(app, parent).exec()`` - see the module docstring."""

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self._worker: Optional[_ValidateWorker] = None
        self.setWindowTitle("Settings")
        self.resize(680, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_claude_group())
        layout.addWidget(self._build_providers_group(), 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._refresh_storage_label()
        self._refresh_providers_table()

    # ==================================================================
    # Claude (Anthropic)
    # ==================================================================
    def _build_claude_group(self) -> QGroupBox:
        box = QGroupBox("Claude (Anthropic)")
        outer = QVBoxLayout(box)

        entry_row = QHBoxLayout()
        self.key_field = QLineEdit()
        self.key_field.setEchoMode(QLineEdit.Password)
        self.key_field.setPlaceholderText(_KEY_PREFIX + "...")
        entry_row.addWidget(self.key_field, 1)
        self.show_key = QCheckBox("Show")
        self.show_key.toggled.connect(self._on_show_toggled)
        entry_row.addWidget(self.show_key)
        outer.addLayout(entry_row)

        self.storage_label = QLabel("-")
        self.storage_label.setObjectName("hint")
        outer.addWidget(self.storage_label)

        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self._on_validate)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.validate_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.validate_label = QLabel("")
        self.validate_label.setWordWrap(True)
        outer.addWidget(self.validate_label)
        return box

    def _on_show_toggled(self, checked: bool) -> None:
        self.key_field.setEchoMode(QLineEdit.Normal if checked
                                   else QLineEdit.Password)

    def _on_save(self) -> None:
        key = self.key_field.text().strip()
        if not key:
            QMessageBox.warning(self, "Settings", "Enter a key first.")
            return
        if not _looks_like_a_claude_key(key):
            QMessageBox.warning(
                self, "Settings",
                "That doesn't look like an Anthropic key - it should start "
                f"with '{_KEY_PREFIX}'.")
            return
        SecretStore().set(ANTHROPIC_KEY_NAME, key)
        self.key_field.clear()
        self.validate_label.setText("")
        self._refresh_router()
        self._refresh_storage_label()
        self._refresh_providers_table()
        self.app.status("Anthropic key saved.")

    def _on_remove(self) -> None:
        if QMessageBox.question(
                self, "Remove key",
                "Remove the stored Anthropic key? Claude will go back to "
                "\"Not configured\" until a new one is entered.",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        SecretStore().delete(ANTHROPIC_KEY_NAME)
        self.validate_label.setText("")
        self._refresh_router()
        self._refresh_storage_label()
        self._refresh_providers_table()
        self.app.status("Anthropic key removed.")

    def _on_validate(self) -> None:
        key = self.key_field.text().strip() or SecretStore().get(ANTHROPIC_KEY_NAME)
        if not key:
            QMessageBox.warning(self, "Settings", "Enter a key first.")
            return
        self.validate_btn.setEnabled(False)
        self.validate_label.setText("Validating...")
        self._worker = _ValidateWorker(key, self)
        self._worker.succeeded.connect(self._on_validate_done)
        self._worker.failed.connect(self._on_validate_done)
        self._worker.finished.connect(lambda: self.validate_btn.setEnabled(True))
        self._worker.start()

    def _on_validate_done(self, message: str) -> None:
        self.validate_label.setText(message)

    def _refresh_router(self) -> None:
        llm = self.app.providers.llm
        if isinstance(llm, RoutedLLM):
            llm.refresh()

    def _refresh_storage_label(self) -> None:
        source = SecretStore().source(ANTHROPIC_KEY_NAME)
        self.storage_label.setText(
            f"Stored in: {STORAGE_LABELS.get(source, 'Not configured')}")
        from_env = source == "environment"
        self.key_field.setEnabled(not from_env)
        self.save_btn.setEnabled(not from_env)
        self.remove_btn.setEnabled(not from_env and source != "none")
        if from_env:
            self.storage_label.setText(
                self.storage_label.text() + " - change it with setx/unset, "
                "not here.")

    # ==================================================================
    # Providers
    # ==================================================================
    def _build_providers_group(self) -> QGroupBox:
        box = QGroupBox("Providers")
        outer = QVBoxLayout(box)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Routing policy:"))
        self.routing_combo = QComboBox()
        self.routing_combo.addItems(list(POLICIES))
        current = self.app.settings.llm_routing
        # Set the initial value before wiring the signal: setCurrentText can
        # fire currentTextChanged synchronously, and the handler touches
        # self.table, which does not exist until later in this method.
        self.routing_combo.setCurrentText(current if current in POLICIES else "auto")
        self.routing_combo.currentTextChanged.connect(self._on_routing_changed)
        top_row.addWidget(self.routing_combo)
        top_row.addStretch(1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_providers_table)
        top_row.addWidget(refresh_btn)
        outer.addLayout(top_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Provider", "Kind", "State", "Model", "Detail"])
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        outer.addWidget(self.table, 1)
        return box

    def _on_routing_changed(self, value: str) -> None:
        if value not in POLICIES:
            return
        self.app.settings.llm_routing = value
        self.app.settings.save()
        llm = self.app.providers.llm
        if isinstance(llm, RoutedLLM):
            llm.policy = value
        self._refresh_providers_table()

    def _refresh_providers_table(self) -> None:
        self._refresh_router()
        rows = self.app.provider_statuses()
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, value in enumerate((r.name, r.kind, r.state, r.model, r.detail)):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)
        self._refresh_storage_label()
