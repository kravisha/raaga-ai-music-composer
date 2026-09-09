"""Voice panel (spec section 14F), and the creator's own takes."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSlider, QTextEdit, QVBoxLayout,
                               QWidget)

from ...voice.renderer import STYLE_PRESETS


class VoicePanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        self.voice_box = QComboBox()
        self.voice_box.activated.connect(self._voice_chosen)
        self.style_box = QComboBox()
        self.style_box.addItems(sorted(STYLE_PRESETS))
        self.style_box.activated.connect(lambda _: self._apply_direction())

        self.sliders = {}
        form = QFormLayout()
        form.addRow("Singer", self.voice_box)
        form.addRow("Delivery", self.style_box)
        for key, label in (("intensity", "Intensity"), ("dynamics", "Dynamics"),
                           ("vibrato", "Vibrato"), ("breath", "Breath"),
                           ("sustain", "Sustained notes"),
                           ("phrase_emphasis", "Phrase emphasis")):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.sliderReleased.connect(self._apply_direction)
            self.sliders[key] = slider
            form.addRow(label, slider)

        self.preview_btn = QPushButton("Render vocal preview")
        self.preview_btn.clicked.connect(lambda: self.app.render_vocal("preview"))
        self.master_btn = QPushButton("Studio vocal-only master")
        self.master_btn.setObjectName("primary")
        self.master_btn.clicked.connect(lambda: self.app.render_vocal("master"))
        self.play_btn = QPushButton("Play vocal")
        self.play_btn.clicked.connect(self._play_vocal)
        self.profile_btn = QPushButton("New voice from recordings...")
        self.profile_btn.clicked.connect(self._create_profile)

        # -- your own takes: Record, Stop, Cancel, and a state that cannot
        # be missed.  The input is open only between Record and Stop.
        self.record_btn = QPushButton("Record a take")
        self.record_btn.clicked.connect(self._record)
        self.stop_take_btn = QPushButton("Stop")
        self.stop_take_btn.clicked.connect(self._stop_take)
        self.cancel_take_btn = QPushButton("Cancel")
        self.cancel_take_btn.clicked.connect(self._cancel_take)
        # -- saved takes: a list you choose from, by id.  Choosing plays
        # nothing and makes nothing; the two buttons below do, and say so.
        self.takes_list = QListWidget()
        self.takes_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.takes_list.setMaximumHeight(96)
        self.takes_list.itemSelectionChanged.connect(self._show_recording_state)
        self.play_take_btn = QPushButton("Play selected take")
        self.play_take_btn.clicked.connect(self._play_selected_take)
        self.profile_from_takes_btn = QPushButton("New voice from selected takes...")
        self.profile_from_takes_btn.clicked.connect(self._create_profile_from_takes)
        self._project_id = ""
        self.recording_state = QLabel("Not recording")
        self.recording_state.setObjectName("recordingState")
        self._clock = QTimer(self)
        self._clock.setInterval(500)
        self._clock.timeout.connect(self._tick)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(110)

        buttons = QHBoxLayout()
        for w in (self.preview_btn, self.master_btn, self.play_btn):
            buttons.addWidget(w)
        buttons.addStretch(1)

        take_buttons = QHBoxLayout()
        for w in (self.record_btn, self.stop_take_btn, self.cancel_take_btn,
                  self.play_take_btn):
            take_buttons.addWidget(w)
        take_buttons.addStretch(1)

        layout = QVBoxLayout(self)
        holder = QWidget()
        holder.setLayout(form)
        layout.addWidget(holder)
        layout.addLayout(buttons)
        layout.addWidget(self.profile_btn)
        layout.addLayout(take_buttons)
        layout.addWidget(self.recording_state)
        layout.addWidget(QLabel("Your saved takes (choose one or more):"))
        layout.addWidget(self.takes_list)
        layout.addWidget(self.profile_from_takes_btn)
        layout.addWidget(QLabel("Rendered vocal takes:"))
        layout.addWidget(self.info, 1)
        self.setMinimumWidth(480)
        self.setMinimumHeight(300)
        self.refresh()

    # -- actions -----------------------------------------------------------
    def _voice_chosen(self, index: int) -> None:
        profile_id = self.voice_box.itemData(index)
        if profile_id:
            self.app.set_voice(str(profile_id))
            self.changed.emit()

    def _apply_direction(self) -> None:
        self.app.set_vocal_direction(
            style=self.style_box.currentText(),
            **{k: s.value() / 100.0 for k, s in self.sliders.items()})
        self.changed.emit()

    def _play_vocal(self) -> None:
        # The controller decides which take is current; the window asking
        # for a master first is what let a stale one outrank a fresh
        # preview.
        self.app.play_vocal()

    def _create_profile(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose your own recordings (WAV)", str(Path.home()),
            "Audio (*.wav *.flac *.ogg)")
        if not paths:
            return
        name, ok = QInputDialog.getText(self, "Voice profile", "Name this voice:")
        if not ok or not name.strip():
            return
        try:
            profile = self.app.create_voice_from_recordings(paths, name.strip())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Voice profile", str(exc))
            return
        QMessageBox.information(self, "Voice profile",
                                f"Created {profile.name}.\n{profile.notes}")
        self.refresh()
        self.changed.emit()

    # -- takes -------------------------------------------------------------
    def _record(self) -> None:
        if self.app.start_take():
            self._clock.start()
        self.refresh()

    def _stop_take(self) -> None:
        self._clock.stop()
        self.app.stop_take()
        self.refresh()
        self.changed.emit()

    def _cancel_take(self) -> None:
        self._clock.stop()
        self.app.cancel_take()
        self.refresh()

    def _selected_take_ids(self) -> list:
        """The ids of the takes chosen in the list, in list order."""
        return [item.data(Qt.UserRole) for item in self.takes_list.selectedItems()
                if item.data(Qt.UserRole)]

    def _play_selected_take(self) -> None:
        chosen = self._selected_take_ids()
        if not chosen:
            self.app.status("Choose a saved take in the list first.")
            return
        if len(chosen) > 1:
            self.app.status("Choose one take to play.")
            return
        # By id, never "the latest": the one chosen is the one played.
        self.app.play_take(chosen[0])

    def _create_profile_from_takes(self) -> None:
        chosen = self._selected_take_ids()
        try:
            takes = self.app.takes_for_reuse(chosen)
        except ValueError as exc:
            QMessageBox.warning(self, "Voice profile", str(exc))
            return
        name, ok = QInputDialog.getText(
            self, "Voice profile",
            f"Name a new voice profile measured from {len(takes)} saved take(s).\n"
            f"The takes are references for pitch range, brightness and noise; "
            f"this is not training and does not promise your voice back.")
        if not ok or not name.strip():
            return
        try:
            profile = self.app.create_voice_from_takes(chosen, name.strip())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Voice profile", str(exc))
            return
        QMessageBox.information(
            self, "Voice profile",
            f"Created {profile.name} from {len(takes)} saved take(s); it is now the "
            f"singer for this song.\n{profile.notes}")
        self.refresh()
        self.changed.emit()

    def _tick(self) -> None:
        if not self.app.recorder.recording:
            self._clock.stop()
        self._show_recording_state()

    def _show_recording_state(self) -> None:
        recording = self.app.recorder.recording
        text = self.app.recording_status()
        if recording:
            text = "● " + text.upper()
        self.recording_state.setText(text)
        self.recording_state.setStyleSheet(
            "color: #d02020; font-weight: bold;" if recording else "")
        self.record_btn.setEnabled(not recording and self.app.recorder.available)
        self.record_btn.setToolTip(
            "" if self.app.recorder.available else self.app.recorder.state.error)
        self.stop_take_btn.setEnabled(recording)
        self.cancel_take_btn.setEnabled(recording)
        chosen = self._selected_take_ids()
        self.play_take_btn.setEnabled(not recording and len(chosen) == 1)
        self.profile_from_takes_btn.setEnabled(not recording and bool(chosen))
        self.play_take_btn.setToolTip("Choose one saved take to play." if len(chosen) != 1 else "")

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        project = self.app.project
        current = project.voice_profile_id
        self.voice_box.blockSignals(True)
        self.voice_box.clear()
        for profile in self.app.voices.all():
            self.voice_box.addItem(
                f"{profile.name}  ({profile.gender}, MIDI "
                f"{profile.range_low}-{profile.range_high})", profile.id)
        idx = self.voice_box.findData(current)
        if idx >= 0:
            self.voice_box.setCurrentIndex(idx)
        self.voice_box.blockSignals(False)

        direction = project.vocal_direction
        self.style_box.blockSignals(True)
        self.style_box.setCurrentText(direction.style)
        self.style_box.blockSignals(False)
        for key, slider in self.sliders.items():
            if not slider.isSliderDown():
                slider.blockSignals(True)
                slider.setValue(int(getattr(direction, key, 0.5) * 100))
                slider.blockSignals(False)

        self.preview_btn.setEnabled(project.melody() is not None)
        self.master_btn.setEnabled(project.melody() is not None)
        self._show_recording_state()

        rows = []
        for take in project.vocal_renders[-12:]:
            profile = self.app.voices.get(take.voice_profile_id)
            marker = " <- master" if take.id == project.vocal_master_id else ""
            rows.append(f"v{take.version} {take.kind:<8} "
                        f"{profile.name if profile else '?':<18} "
                        f"{take.duration:.0f}s  {take.direction.style}{marker}")
        self.info.setPlainText("\n".join(rows) or "No rendered vocal takes yet.")

        # The saved takes, every one, keyed by id.  A selection survives an
        # ordinary refresh by id; a new song starts with none, so a take of
        # the last song is never the one chosen in this one.
        keep = set(self._selected_take_ids()) if project.project_id == self._project_id else set()
        self._project_id = project.project_id
        self.takes_list.blockSignals(True)
        self.takes_list.clear()
        for take in project.recordings:
            when = time.strftime("%H:%M", time.localtime(take.created_at))
            where = take.section_name or "whole song"
            item = QListWidgetItem(f"{take.label}  -  {where}  -  {take.duration:.1f}s  -  "
                                   f"recorded by you at {when}")
            item.setData(Qt.UserRole, take.id)
            self.takes_list.addItem(item)
            item.setSelected(take.id in keep)
        self.takes_list.blockSignals(False)
        self._show_recording_state()
