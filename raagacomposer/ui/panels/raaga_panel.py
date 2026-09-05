"""Raaga panel (spec section 14C)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QTextEdit, QVBoxLayout)

from ...core.actions import ActionState
from ...raaga.selection import compare


class RaagaPanel(QGroupBox):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__("Raaga", parent)
        self.app = app

        self.suggestions = QListWidget()
        self.suggestions.setFixedHeight(96)
        self.suggestions.currentRowChanged.connect(self._show_details)
        self.suggestions.itemDoubleClicked.connect(lambda _: self.accept_selected())

        self.all_raagas = QComboBox()
        self.all_raagas.addItems(self.app.raagas.names())
        self.all_raagas.currentTextChanged.connect(self._show_named)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setFixedHeight(132)

        self.selected_label = QLabel("Selected: -")
        self.lock_box = QCheckBox("Lock the raaga")
        self.lock_box.toggled.connect(self._toggle_lock)

        suggest_btn = QPushButton("Suggest from the brief")
        suggest_btn.clicked.connect(self.suggest)
        accept_btn = QPushButton("Use this raaga")
        accept_btn.setObjectName("primary")
        accept_btn.clicked.connect(self.accept_selected)
        compare_btn = QPushButton("Compare with selected")
        compare_btn.clicked.connect(self.compare_with_current)
        # The pack's audition step (document 05 section 7): hear the scale
        # before composing in it.  A ranked list with a reason attached is an
        # argument; the arohanam and avarohanam played are the evidence.
        self.audition_btn = QPushButton("Hear the scale")
        self.audition_btn.setToolTip(
            "Play this raaga's arohanam and avarohanam, exactly as the "
            "library stores them.")
        self.audition_btn.clicked.connect(self.audition_selected)
        # Saying no is a training signal the pack asks us to learn from
        # (Stage 1 pack document 05 section 6); without a control for it the
        # creator could only ever teach the agent by agreeing with it.
        self.reject_btn = QPushButton("Not this one")
        self.reject_btn.setToolTip(
            "Rank this raaga lower for briefs like this one. "
            "The raaga itself is unchanged.")
        self.reject_btn.clicked.connect(self.reject_selected)

        # Two to a row: this panel is a narrow column, and four controls
        # across it clip the last one out of reach.
        row1 = QHBoxLayout()
        row1.addWidget(suggest_btn)
        row1.addWidget(accept_btn)
        row2 = QHBoxLayout()
        row2.addWidget(self.audition_btn)
        row2.addWidget(self.reject_btn)
        row3 = QHBoxLayout()
        row3.addWidget(compare_btn)
        row3.addWidget(self.lock_box)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Suggested for this brief:"))
        layout.addWidget(self.suggestions)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(QLabel("Or choose any raaga:"))
        layout.addWidget(self.all_raagas)
        layout.addLayout(row3)
        layout.addWidget(self.details)
        layout.addWidget(self.selected_label)
        self.refresh()

        # Chain onto whatever is already listening (spec section 6.1): Apply
        # Brief now ranks suggestions itself, so this panel refreshes as soon
        # as that action completes instead of waiting for a second click on
        # "Suggest from the brief".
        previous_on_action = self.app.on_action

        def _on_action(status) -> None:
            if previous_on_action:
                previous_on_action(status)
            if status.action == "apply_brief" and status.state == ActionState.COMPLETED:
                self._render_suggestions(self.app.last_suggestions)

        self.app.on_action = _on_action

    # -- actions -----------------------------------------------------------
    def suggest(self) -> None:
        self._render_suggestions(self.app.raaga_suggestions())

    def _render_suggestions(self, suggestions) -> None:
        self.suggestions.clear()
        for s in suggestions:
            reason = getattr(s, "reason", "") or getattr(s, "rationale", "")
            confidence = getattr(s, "confidence", None)
            label = f"{s.name}  -  {reason}"
            if confidence is not None:
                label += f"  (confidence {float(confidence):.2f})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, s.name)
            self.suggestions.addItem(item)
        if self.suggestions.count():
            self.suggestions.setCurrentRow(0)
        self.changed.emit()

    def _current_name(self) -> str:
        item = self.suggestions.currentItem()
        if item is not None:
            return str(item.data(Qt.UserRole))
        return self.all_raagas.currentText()

    def accept_selected(self) -> None:
        name = self._current_name()
        if not name:
            return
        try:
            self.app.select_raaga(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Raaga", str(exc))
            return
        self.refresh()
        self.changed.emit()

    def audition_selected(self) -> None:
        """Play the highlighted raaga's scale, whichever way it was picked."""
        name = self._current_name()
        if not name:
            return
        try:
            self.app.audition_raaga(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Raaga", str(exc))

    def reject_selected(self) -> None:
        """Turn a suggestion down, and re-rank with that taken into account."""
        name = self._current_name()
        if not name:
            return
        try:
            self.app.reject_raaga(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Raaga", str(exc))
            return
        # Ask again straight away, so the creator sees the effect of what
        # they just said rather than having to press Suggest themselves.
        self.suggest()

    def compare_with_current(self) -> None:
        other = self.app.raagas.get(self._current_name())
        current = self.app.current_raaga()
        if other is None:
            return
        if current is None or current.name == other.name:
            self.details.setPlainText(other.describe())
            return
        self.details.setPlainText(compare(current, other))

    def _toggle_lock(self, checked: bool) -> None:
        if self.app.project.raaga.locked != checked:
            self.app.set_raaga_lock(checked)
            self.changed.emit()

    def _show_details(self, row: int) -> None:
        if row < 0:
            return
        raaga = self.app.raagas.get(str(self.suggestions.item(row).data(Qt.UserRole)))
        if raaga:
            self.details.setPlainText(raaga.describe())

    def _show_named(self, name: str) -> None:
        raaga = self.app.raagas.get(name)
        if raaga:
            self.details.setPlainText(raaga.describe())

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        choice = self.app.project.raaga
        self.selected_label.setText(
            f"Selected: {choice.selected or '-'}"
            f"{'  [locked]' if choice.locked else ''}")
        self.lock_box.blockSignals(True)
        self.lock_box.setChecked(choice.locked)
        self.lock_box.blockSignals(False)
        if choice.selected and not self.details.toPlainText():
            raaga = self.app.current_raaga()
            if raaga:
                self.details.setPlainText(raaga.describe())
