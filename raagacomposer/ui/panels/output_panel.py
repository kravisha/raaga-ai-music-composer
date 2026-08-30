"""Output and export panel (spec sections 14I, 19)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QScrollArea,
                               QTextEdit, QVBoxLayout, QWidget)


class OutputPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        render_box = QGroupBox("Render")
        grid = QGridLayout(render_box)
        actions = [
            ("Vocal only (studio master)", lambda: self.app.render_vocal("master")),
            ("Instrumental only", lambda: self.app.render("instrumental")),
            ("Full mix", lambda: self.app.render("full")),
            ("Tune preview", lambda: self.app.render("tune")),
        ]
        for i, (label, fn) in enumerate(actions):
            b = QPushButton(label)
            if i == 2:
                b.setObjectName("primary")
            b.clicked.connect(fn)
            grid.addWidget(b, i // 2, i % 2)

        play_box = QGroupBox("Play")
        play_row = QHBoxLayout(play_box)
        for label, kind in (("Vocal master", "vocal_master"),
                            ("Vocal preview", "vocal_preview"),
                            ("Instrumental", "instrumental"),
                            ("Full mix", "full"),
                            ("Tune", "tune")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.app.play_render(k))
            play_row.addWidget(b)

        export_box = QGroupBox("Export")
        egrid = QGridLayout(export_box)
        exports = [
            ("Full mix (WAV/MP3)", lambda: self._export_audio("full")),
            ("Vocal only (WAV/MP3)", lambda: self._export_audio("vocal_master")),
            ("Instrumental (WAV/MP3)", lambda: self._export_audio("instrumental")),
            ("Stems folder", self._export_stems),
            ("MIDI", self._export_midi),
            ("MusicXML", self._export_musicxml),
            ("Lyrics (text)", self._export_lyrics),
            ("Project archive (zip)", self._archive),
        ]
        for i, (label, fn) in enumerate(exports):
            b = QPushButton(label)
            b.clicked.connect(fn)
            egrid.addWidget(b, i // 2, i % 2)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(120)

        # The groups keep their natural height and the panel scrolls, rather
        # than squeezing the buttons until their labels are unreadable.
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(render_box)
        inner_layout.addWidget(play_box)
        inner_layout.addWidget(export_box)
        inner_layout.addWidget(QLabel("Current outputs:"))
        inner_layout.addWidget(self.info, 1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setMinimumWidth(480)
        self.setMinimumHeight(240)
        self.refresh()

    # -- exports -----------------------------------------------------------
    def _default_dir(self) -> str:
        return str(self.app.project_dir / "exports" if self.app.project_dir
                   else Path.home())

    def _export_audio(self, kind: str) -> None:
        if self.app.rendered(kind) is None:
            QMessageBox.information(self, "Export",
                                    f"Render the {kind.replace('_', ' ')} first.")
            return
        suggested = f"{self.app.project.title} - {kind}.wav"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", str(Path(self._default_dir()) / suggested),
            "WAV (*.wav);;MP3 (*.mp3)")
        if not path:
            return
        out = self.app.export(Path(path), kind)
        if out:
            QMessageBox.information(self, "Export", f"Written to {out}")
        self.refresh()

    def _export_stems(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export stems into",
                                                     self._default_dir())
        if directory:
            files = self.app.export_stems(Path(directory))
            QMessageBox.information(self, "Export", f"Wrote {len(files)} stem(s).")

    def _export_midi(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI",
            str(Path(self._default_dir()) / f"{self.app.project.title}.mid"),
            "MIDI (*.mid)")
        if path:
            self.app.export_midi(Path(path))

    def _export_musicxml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export notation",
            str(Path(self._default_dir()) / f"{self.app.project.title}.musicxml"),
            "MusicXML (*.musicxml)")
        if path:
            self.app.export_musicxml(Path(path))

    def _export_lyrics(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export lyrics",
            str(Path(self._default_dir()) / f"{self.app.project.title}.txt"),
            "Text (*.txt)")
        if path:
            self.app.export_lyrics(Path(path))

    def _archive(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Archive project",
            str(Path.home() / f"{self.app.project.title}.zip"), "Zip (*.zip)")
        if path:
            out = self.app.archive(Path(path))
            QMessageBox.information(self, "Archive", f"Archived to {out}")

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        rows = []
        for kind in ("tune", "vocal_preview", "vocal_master", "instrumental", "full"):
            rendered = self.app.rendered(kind)
            if rendered is None:
                rows.append(f"{kind:<15} - not rendered")
            else:
                rows.append(f"{kind:<15} {rendered.duration:6.1f}s  "
                            f"{Path(rendered.path).name if rendered.path else ''}")
        rows.append("")
        rows.append(self.app.summary())
        rows.append("")
        rows.append("Providers:")
        rows.append(self.app.providers.summary())
        for note in self.app.providers.notes:
            rows.append(f"  note: {note}")
        self.info.setPlainText("\n".join(rows))
