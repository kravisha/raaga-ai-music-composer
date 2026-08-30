"""Voice panel (spec section 14F)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QMessageBox, QPushButton,
                               QSlider, QTextEdit, QVBoxLayout, QWidget)

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

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setFixedHeight(110)

        buttons = QHBoxLayout()
        for w in (self.preview_btn, self.master_btn, self.play_btn):
            buttons.addWidget(w)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        holder = QWidget()
        holder.setLayout(form)
        layout.addWidget(holder)
        layout.addLayout(buttons)
        layout.addWidget(self.profile_btn)
        layout.addWidget(QLabel("Takes:"))
        layout.addWidget(self.info, 1)
        self.setMinimumWidth(480)
        self.setMinimumHeight(260)
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
        for kind in ("vocal_master", "vocal_preview"):
            if self.app.rendered(kind) is not None:
                self.app.play_render(kind)
                return
        self.app.status("Render a vocal take first.")

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

        rows = []
        for take in project.vocal_renders[-12:]:
            profile = self.app.voices.get(take.voice_profile_id)
            marker = " <- master" if take.id == project.vocal_master_id else ""
            rows.append(f"v{take.version} {take.kind:<8} "
                        f"{profile.name if profile else '?':<18} "
                        f"{take.duration:.0f}s  {take.direction.style}{marker}")
        self.info.setPlainText("\n".join(rows) or "No vocal takes yet.")
