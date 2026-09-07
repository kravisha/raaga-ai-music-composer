"""Raaga panel (spec section 14C)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSizePolicy, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)

from ...core.actions import ActionState
from ...raaga.selection import compare


class RaagaPanel(QGroupBox):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__("Raaga", parent)
        self.app = app

        #: Which control the creator last used - "suggestions" or "all".
        self._picked_from = "suggestions"
        #: Why the current brief suggested each raaga: {name: (reason,
        #: confidence)}.  Emptied and refilled whenever suggestions are
        #: rendered, so it never describes a brief that has moved on.
        self._rationales: dict = {}
        #: The ranking these rationales came from, and the brief it ranked
        #: for.  A reason is only evidence while the ranking that produced
        #: it is still the current one.
        self._rendered_epoch = None
        self._rendered_brief = None

        self.suggestions = QListWidget()
        self.suggestions.setMinimumHeight(64)
        # Ignored vertically: a list's size hint grows with its contents, and
        # inside a splitter that let four suggestions push the whole left
        # column past the height that keeps controls above the fold.  The
        # splitter decides how tall this is; the list does not get a vote.
        self.suggestions.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.suggestions.currentRowChanged.connect(self._show_details)
        self.suggestions.itemDoubleClicked.connect(lambda _: self.accept_selected())

        # Every raaga the library knows, whether the brief suggested it or
        # not, and searchable because there are ninety-three of them.
        self.all_raagas = QComboBox()
        self.all_raagas.setEditable(True)
        self.all_raagas.setInsertPolicy(QComboBox.NoInsert)
        self.all_raagas.addItems(self.app.raagas.names())
        self.all_raagas.setCurrentIndex(-1)
        self.all_raagas.lineEdit().setPlaceholderText("Search any raaga...")
        completer = self.all_raagas.completer()
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.all_raagas.currentTextChanged.connect(self._show_named)
        # ``activated`` fires only when a person chooses, where
        # ``currentTextChanged`` also fires when the list is rebuilt.  The
        # difference matters: refreshing the panel must not look like a
        # choice.
        self.all_raagas.activated.connect(lambda _: self._chose("all"))
        # Both, deliberately.  ``currentRowChanged`` does not fire when the
        # clicked row is already current, so clicking the highlighted
        # suggestion refreshed nothing and looked like the details had
        # stopped updating.
        self.suggestions.itemClicked.connect(self._on_suggestion_clicked)

        # Hear any raaga, straight from the selector, with no ambiguity about
        # which one is meant.  ``Hear the scale`` acts on whichever control
        # was last used; this one always means the box beside it.
        self.preview_btn = QPushButton("Play this scale")
        self.preview_btn.setToolTip(
            "Play the scale of the raaga chosen in the box, whether or not "
            "the brief suggested it. Does not change the song's raaga.")
        self.preview_btn.clicked.connect(self.preview_chosen)

        self.use_chosen_btn = QPushButton("Use this one")
        self.use_chosen_btn.setToolTip(
            "Compose in the raaga chosen in the box.")
        self.use_chosen_btn.clicked.connect(self.use_chosen)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        # A modest minimum, because the splitter below is what makes this
        # big now.  Insisting on 132 pixels here pushed the left column's
        # size hint to 910 and tripped the guard that keeps controls above
        # the fold - the drag handle does the work a large minimum used to.
        self.details.setMinimumHeight(96)
        self.details.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.details.setLineWrapMode(QTextEdit.WidgetWidth)

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

        # A drag handle between the list and the details, because a size
        # policy is not an affordance: measured in the real left column it
        # gave the details 139 pixels at a 900-pixel window while the text
        # still scrolled by 389, which is an improvement and not the thing
        # Krish asked for.  A splitter lets him make the details as tall as
        # he wants, and remembers where he put it for the session.
        self._split = QSplitter(Qt.Vertical)
        self._split.setChildrenCollapsible(False)
        self._split.setHandleWidth(6)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(QLabel("Suggested for this brief:"))
        top_layout.addWidget(self.suggestions, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(QLabel("Or choose any raaga:"))
        chooser = QHBoxLayout()
        chooser.addWidget(self.all_raagas, 1)
        chooser.addWidget(self.preview_btn)
        chooser.addWidget(self.use_chosen_btn)
        layout.addLayout(chooser)
        layout.addLayout(row3)
        self._split.addWidget(top)
        self._split.addWidget(self.details)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.setSizes([110, 340])
        layout.insertWidget(0, self._split, 1)
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
        self._rationales = {}
        for s in suggestions:
            reason = getattr(s, "reason", "") or getattr(s, "rationale", "")
            confidence = getattr(s, "confidence", None)
            label = f"{s.name}  -  {reason}"
            if confidence is not None:
                label += f"  (confidence {float(confidence):.2f})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, s.name)
            # The reason *this* brief gave for *this* raaga, kept beside the
            # name.  Only the name was stored, so the details pane and the
            # comparison - which look a raaga up in the catalogue - could
            # show what the raaga is but never why it had been suggested.
            # That reason exists once, at ranking time; recomputing it to
            # display it would re-run the recommendation as a side effect of
            # looking at it.
            item.setData(Qt.UserRole + 1, reason)
            if confidence is not None:
                item.setData(Qt.UserRole + 2, float(confidence))
            self._rationales[s.name] = (reason, confidence)
            self.suggestions.addItem(item)
        self._rendered_epoch = getattr(self.app, "suggestion_epoch", None)
        self._rendered_brief = getattr(self.app, "suggested_for", None)
        if self.suggestions.count():
            self.suggestions.setCurrentRow(0)
        self.changed.emit()

    def _on_suggestion_clicked(self, item) -> None:
        self._chose("suggestions")
        self._show_named(str(item.data(Qt.UserRole)))

    def _chose(self, where: str) -> None:
        """Remember which control the creator last used."""
        self._picked_from = where

    def _current_name(self) -> str:
        """The raaga the creator last pointed at, from either control.

        This used to read the suggestions list first and fall back to the
        box only when nothing was highlighted.  Applying a brief always
        highlights a row, so from then on the box was unreachable: choosing
        Keeravani there and pressing "Hear the scale" played whichever
        suggestion happened to be selected.  The creator could see the
        raaga they had picked and could not hear it.

        The control that was last used decides, which is the same rule the
        audition fix needed in the morning: what is acted on has to be what
        the creator last touched.
        """
        if self._picked_from == "all":
            chosen = self.all_raagas.currentText().strip()
            if chosen:
                return chosen
        item = self.suggestions.currentItem()
        if item is not None:
            return str(item.data(Qt.UserRole))
        return self.all_raagas.currentText().strip()

    def preview_chosen(self) -> None:
        """Hear the scale of whatever is in the box - never a suggestion."""
        name = self.all_raagas.currentText().strip()
        if not name:
            QMessageBox.information(self, "Raaga",
                                    "Choose a raaga in the box first.")
            return
        self._chose("all")
        try:
            self.app.audition_raaga(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Raaga", str(exc))

    def use_chosen(self) -> None:
        """Compose in whatever is in the box - never a suggestion."""
        name = self.all_raagas.currentText().strip()
        if not name:
            QMessageBox.information(self, "Raaga",
                                    "Choose a raaga in the box first.")
            return
        self._chose("all")
        self.accept_selected()

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
            self.details.setPlainText(self._describe_with_reason(other.name))
            return
        # The catalogue comparison, followed by whatever this brief said
        # about either raaga - the reason a raaga was put in front of the
        # creator belongs beside the facts about it, not only in the list
        # row it was clipped into.
        text = compare(current, other)
        notes = [n for n in (self._recommendation_note(current.name),
                             self._recommendation_note(other.name)) if n]
        if notes:
            text += "\n\n" + "\n\n".join(notes)
        self.details.setPlainText(text)

    def _toggle_lock(self, checked: bool) -> None:
        if self.app.project.raaga.locked != checked:
            self.app.set_raaga_lock(checked)
            self.changed.emit()

    def _show_details(self, row: int) -> None:
        if row < 0:
            return
        name = str(self.suggestions.item(row).data(Qt.UserRole))
        if self.app.raagas.get(name):
            self.details.setPlainText(self._describe_with_reason(name))

    def _recommendation_note(self, name: str) -> str:
        """What this brief said about this raaga, if it suggested it.

        The raaga is named in the heading because the comparison prints two
        of these one after another: "Why this brief suggested it" twice,
        under a comparison of Keeravani with Chitrambari, left the creator
        to guess which explanation belonged to which raaga.
        """
        reason, confidence = self._rationales.get(name, ("", None))
        if not reason:
            return ""
        line = f"Why this brief suggested {name}: {reason}"
        if confidence is not None:
            line += f" (confidence {float(confidence):.2f})"
        return line + "\n(" + self._ranking_provenance() + ", not the catalogue)"

    def _ranking_provenance(self) -> str:
        """Whether the ranking that gave this reason is still the current one.

        The brief can be edited without being applied, and then the reasons
        on screen answer a question that has since changed.  They are still
        worth showing - they are why these raagas are in the list at all -
        but they are not what the brief now in the panel would say.
        """
        project = getattr(self.app, "project", None)
        current = getattr(project, "brief", None)
        moved_on = (self._rendered_brief is not None and current is not None
                    and self._rendered_brief != current)
        if moved_on:
            return "from an earlier brief's ranking"
        return "from the current brief's ranking"

    def _drop_a_ranking_that_is_no_longer_ours(self) -> None:
        """Throw away suggestions that belong to a project we have left.

        The controller discards its ranking when a project is created or
        opened and bumps its epoch.  Without this the list, the reasons and
        the details pane all survived the change, and the previous song's
        recommendation was shown as evidence about the new one.
        """
        epoch = getattr(self.app, "suggestion_epoch", None)
        if self._rendered_epoch is None or epoch == self._rendered_epoch:
            return
        if getattr(self.app, "last_suggestions", None):
            # A new ranking exists; show that one rather than nothing.
            self._render_suggestions(self.app.last_suggestions)
            return
        self.suggestions.clear()
        self._rationales = {}
        self._rendered_epoch = None
        self._rendered_brief = None
        self.details.clear()

    def _describe_with_reason(self, name: str) -> str:
        raaga = self.app.raagas.get(name)
        if raaga is None:
            return f"{name} is not in the library."
        note = self._recommendation_note(name)
        return f"{raaga.describe()}\n\n{note}" if note else raaga.describe()

    def _show_named(self, name: str) -> None:
        if self.app.raagas.get(name):
            self.details.setPlainText(self._describe_with_reason(name))

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        self._drop_a_ranking_that_is_no_longer_ours()
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
