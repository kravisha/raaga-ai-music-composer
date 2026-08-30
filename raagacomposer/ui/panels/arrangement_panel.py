"""Arrangement and timeline panel (spec sections 8, 14G)."""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea, QSizePolicy,
                               QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ...music import instruments as catalog
from ..timeline import TimelineWidget

ROLES = ["auto", "lead", "counter", "pad", "bass", "rhythm", "fill", "drone"]


class ArrangementPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        # ---- timeline ----------------------------------------------------
        self.timeline = TimelineWidget()
        self.timeline.seekRequested.connect(self._seek)
        self.timeline.selectionChanged.connect(self._selection_changed)
        self.timeline.regionSelected.connect(self._region_selected)
        self.timeline.regionAction.connect(self._region_action)
        self.timeline.trackAction.connect(self._track_action)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.timeline)
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # The timeline is deliberately wider than the window; the scroll area
        # must not push that width back into the layout.
        self.scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.scroll.setMinimumHeight(150)

        # ---- add / replace controls --------------------------------------
        self.instrument_box = QComboBox()
        for inst in catalog.all_instruments():
            self.instrument_box.addItem(f"{inst.name}  ({inst.family})", inst.key)
        self.role_box = QComboBox()
        self.role_box.addItems(ROLES)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 3600.0)
        self.start_spin.setSuffix(" s")
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 3600.0)
        self.end_spin.setSuffix(" s")
        self.whole_box = QCheckBox("whole song")
        self.whole_box.setChecked(True)

        self.intensity = QSlider(Qt.Horizontal)
        self.intensity.setRange(10, 100)
        self.intensity.setValue(60)
        self.intensity.setFixedWidth(90)

        add_btn = QPushButton("Add")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove)
        replace_btn = QPushButton("Replace with...")
        replace_btn.clicked.connect(self._replace)
        auto_btn = QPushButton("Auto-arrange")
        auto_btn.clicked.connect(lambda: self.app.auto_arrange())

        self.feel_edit = QLineEdit()
        self.feel_edit.setPlaceholderText(
            "...or describe a feel: \"lonely, late at night, but still warm\"")
        feel_btn = QPushButton("Suggest instrument")
        feel_btn.clicked.connect(self._suggest)

        self.instrument_box.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.instrument_box.setMinimumContentsLength(12)
        self.feel_edit.setMinimumWidth(150)
        for spin in (self.start_spin, self.end_spin):
            spin.setMaximumWidth(90)

        controls = QGroupBox("Instruments")
        c1 = QHBoxLayout()
        c1.addWidget(QLabel("Instrument"))
        c1.addWidget(self.instrument_box, 2)
        c1.addWidget(QLabel("Role"))
        c1.addWidget(self.role_box)
        c1.addWidget(QLabel("Density"))
        c1.addWidget(self.intensity)
        c2 = QHBoxLayout()
        c2.addWidget(QLabel("From"))
        c2.addWidget(self.start_spin)
        c2.addWidget(QLabel("To"))
        c2.addWidget(self.end_spin)
        c2.addWidget(self.whole_box)
        c2.addWidget(add_btn)
        c2.addWidget(remove_btn)
        c2.addWidget(replace_btn)
        c2.addWidget(auto_btn)
        c3 = QHBoxLayout()
        c3.addWidget(self.feel_edit, 2)
        c3.addWidget(feel_btn)
        cv = QVBoxLayout(controls)
        cv.addLayout(c1)
        cv.addLayout(c2)
        cv.addLayout(c3)

        # ---- track table -------------------------------------------------
        self.tracks = QTableWidget(0, 7)
        self.tracks.setHorizontalHeaderLabels(
            ["Track", "Role", "Regions", "Mute", "Solo", "Lock", "Gain"])
        self.tracks.verticalHeader().setVisible(False)
        self.tracks.setFixedHeight(112)
        self.tracks.cellClicked.connect(self._cell_clicked)

        zoom_in = QPushButton("Zoom in")
        zoom_in.clicked.connect(lambda: self.timeline.zoom(1.25))
        zoom_out = QPushButton("Zoom out")
        zoom_out.clicked.connect(lambda: self.timeline.zoom(1 / 1.25))
        fit = QPushButton("Fit")
        fit.clicked.connect(
            lambda: self.timeline.zoom_to_fit(self.scroll.viewport().width()))
        lock_sel = QPushButton("Lock selection")
        lock_sel.clicked.connect(lambda: self._lock_selection(True))
        unlock_sel = QPushButton("Unlock selection")
        unlock_sel.clicked.connect(lambda: self._lock_selection(False))
        play_sel = QPushButton("Play selection")
        play_sel.clicked.connect(self._play_selection)

        zoom_row = QHBoxLayout()
        for w in (zoom_in, zoom_out, fit, play_sel, lock_sel, unlock_sel):
            zoom_row.addWidget(w)
        self.selection_label = QLabel("No selection")
        self.selection_label.setObjectName("hint")
        zoom_row.addWidget(self.selection_label, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addLayout(zoom_row)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self.tracks)
        # Panels must not dictate the window width; inner views scroll instead.
        self.setMinimumWidth(560)
        self.setMinimumHeight(320)
        self.tracks.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.refresh()

    # -- helpers -----------------------------------------------------------
    def _range(self) -> Tuple[float, float]:
        if self.whole_box.isChecked():
            return 0.0, max(1.0, self.app.project.duration)
        return float(self.start_spin.value()), float(self.end_spin.value())

    def _instrument(self) -> str:
        return str(self.instrument_box.currentData())

    def _seek(self, seconds: float) -> None:
        self.app.seek(seconds)
        self.timeline.set_playhead(seconds)

    def _selection_changed(self, selection) -> None:  # noqa: ANN001
        if selection:
            start, end = selection
            self.app.set_selection(start, end)
            self.whole_box.setChecked(False)
            self.start_spin.setValue(start)
            self.end_spin.setValue(end)
            self.selection_label.setText(f"Selection {start:.1f}s - {end:.1f}s")
        else:
            self.app.set_selection(None, None)
            self.selection_label.setText("No selection")

    def _region_selected(self, track_id: str, region_id: str) -> None:
        self.app.context.last_track_id = track_id
        self.app.context.last_region_id = region_id
        arrangement = self.app.project.arrangement()
        track = arrangement.track_by_id(track_id) if arrangement else None
        region = track.region_by_id(region_id) if track else None
        if track and region:
            self.app.context.last_instrument = track.instrument
            idx = self.instrument_box.findData(track.instrument)
            if idx >= 0:
                self.instrument_box.setCurrentIndex(idx)
            self.whole_box.setChecked(False)
            self.start_spin.setValue(region.start)
            self.end_spin.setValue(region.end)
            self.selection_label.setText(
                f"{track.label}: {region.start:.1f}-{region.end:.1f}s "
                f"({region.role}, {len(region.notes)} notes"
                f"{', locked' if region.locked else ''})")

    # -- actions -----------------------------------------------------------
    def _add(self) -> None:
        start, end = self._range()
        role = self.role_box.currentText()
        self.app.add_instrument(self._instrument(), start, end,
                                role="" if role == "auto" else role,
                                intensity=self.intensity.value() / 100.0)
        self.changed.emit()

    def _remove(self) -> None:
        start, end = self._range()
        whole = self.whole_box.isChecked()
        self.app.remove_instrument(self._instrument(),
                                   None if whole else start,
                                   None if whole else end)
        self.changed.emit()

    def _replace(self) -> None:
        names = [i.name for i in catalog.all_instruments()]
        name, ok = QInputDialog.getItem(self, "Replace instrument",
                                        "Replace with:", names, 0, False)
        if not ok:
            return
        target = catalog.find(name)
        if target is None:
            return
        start, end = self._range()
        whole = self.whole_box.isChecked()
        self.app.replace_instrument(self._instrument(), target.key,
                                    None if whole else start,
                                    None if whole else end)
        self.changed.emit()

    def _suggest(self) -> None:
        from ...raaga.selection import expand_feel_words
        text = self.feel_edit.text().strip()
        if not text:
            QMessageBox.information(self, "Instruments",
                                    "Describe the feel you want first.")
            return
        words = expand_feel_words(text) or [text]
        ranked = self.app.suggest_instruments(words)
        if not ranked:
            QMessageBox.information(self, "Instruments",
                                    "Nothing in the catalog matches that feel.")
            return
        options = [f"{i.name}  -  {', '.join(i.tags[:4])}" for i, _ in ranked]
        choice, ok = QInputDialog.getItem(
            self, "Suggested instruments",
            f"For \"{text}\" I would try:", options, 0, False)
        if not ok:
            return
        inst = ranked[options.index(choice)][0]
        start, end = self._range()
        self.app.add_instrument(inst.key, start, end,
                                intensity=self.intensity.value() / 100.0)
        self.changed.emit()

    def _lock_selection(self, locked: bool) -> None:
        start, end = self._range()
        self.app.lock_range(start, end, locked)
        self.changed.emit()

    def _play_selection(self) -> None:
        start, end = self._range()
        self.app.play_range(start, end)

    def _region_action(self, action: str, track_id: str, region_id: str) -> None:
        arrangement = self.app.project.arrangement()
        if arrangement is None:
            return
        track = arrangement.track_by_id(track_id)
        region = track.region_by_id(region_id) if track else None
        if track is None or region is None:
            return
        if action == "regenerate":
            self.app.regenerate_region(track_id, region_id)
        elif action in ("lock", "unlock"):
            region.locked = (action == "lock")
            self.app._changed("region.lock",
                              f"{action.title()}ed {track.label} "
                              f"{region.start:.0f}-{region.end:.0f}s")
        elif action == "remove":
            self.app.remove_instrument(track.instrument, region.start, region.end)
        elif action == "replace":
            self.instrument_box.setCurrentIndex(
                max(0, self.instrument_box.findData(track.instrument)))
            self.whole_box.setChecked(False)
            self.start_spin.setValue(region.start)
            self.end_spin.setValue(region.end)
            self._replace()
        self.changed.emit()

    def _track_action(self, action: str, track_id: str) -> None:
        arrangement = self.app.project.arrangement()
        if action == "add":
            self._add()
            return
        if arrangement is None:
            return
        track = arrangement.track_by_id(track_id)
        if track is None:
            return
        if action == "mute":
            self.app.set_track_flag(track_id, mute=not track.mute)
        elif action == "solo":
            self.app.set_track_flag(track_id, solo=not track.solo)
        elif action == "lock":
            self.app.set_track_flag(track_id, locked=not track.locked)
        elif action == "remove":
            self.app.remove_instrument(track.instrument)
        elif action == "select":
            idx = self.instrument_box.findData(track.instrument)
            if idx >= 0:
                self.instrument_box.setCurrentIndex(idx)
        self.changed.emit()

    def _cell_clicked(self, row: int, column: int) -> None:
        arrangement = self.app.project.arrangement()
        if arrangement is None or row >= len(arrangement.tracks):
            return
        track = arrangement.tracks[row]
        if column == 3:
            self.app.set_track_flag(track.id, mute=not track.mute)
        elif column == 4:
            self.app.set_track_flag(track.id, solo=not track.solo)
        elif column == 5:
            self.app.set_track_flag(track.id, locked=not track.locked)
        else:
            idx = self.instrument_box.findData(track.instrument)
            if idx >= 0:
                self.instrument_box.setCurrentIndex(idx)

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        project = self.app.project
        melody = project.melody()
        arrangement = project.arrangement()
        self.timeline.set_project(melody, arrangement, project.duration)
        kind = self.app.best_render()
        rendered = self.app.rendered(kind) if kind else None
        if rendered is not None:
            self.timeline.set_waveform(rendered.audio, rendered.sample_rate)
        self.timeline.set_selection(self.app.selection)

        if self.whole_box.isChecked():
            self.start_spin.setValue(0.0)
            self.end_spin.setValue(max(1.0, project.duration))

        tracks = arrangement.tracks if arrangement else []
        self.tracks.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            spans = ", ".join(f"{r.start:.0f}-{r.end:.0f}"
                              f"{'*' if r.locked else ''}" for r in track.regions)
            values = [track.label, track.role, spans,
                      "M" if track.mute else "-", "S" if track.solo else "-",
                      "LOCK" if track.locked else "-", f"{track.gain:.2f}"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.tracks.setItem(row, col, item)
        self.tracks.resizeColumnsToContents()

    def update_playhead(self) -> None:
        self.timeline.set_playhead(self.app.playback.position)
