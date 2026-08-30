"""Main desktop window (spec section 14).

A native Qt window: project header, creative brief and raaga down the left; tune,
lyrics, voice and output as tabs in the centre above the arrangement timeline;
the conversation with the microphone down the right; transport and job progress
along the top and bottom.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSplitter,
                               QStatusBar, QTabWidget, QToolBar, QVBoxLayout,
                               QWidget)

from ..app import AppController
from ..core.logging_setup import get_logger
from ..music.theory import format_time
from . import theme
from .panels.arrangement_panel import ArrangementPanel
from .panels.brief_panel import BriefPanel
from .panels.conversation_panel import ConversationPanel
from .panels.lyrics_panel import LyricsPanel
from .panels.output_panel import OutputPanel
from .panels.project_panel import ProjectPanel
from .panels.raaga_panel import RaagaPanel
from .panels.tune_panel import TunePanel
from .panels.voice_panel import VoicePanel

log = get_logger("ui")


class MainWindow(QMainWindow):
    def __init__(self, app: AppController) -> None:
        super().__init__()
        self.app = app
        self.setWindowTitle("Raaga AI Music Composer")
        self.resize(1580, 980)

        self._build_panels()
        self._build_toolbar()
        self._build_menu()
        self._build_status()

        app.on_project_changed = self.refresh
        app.on_status = self._status_message
        app.on_conversation = self.conversation.refresh
        app.on_render = lambda kind: self.arrangement.refresh()
        app.on_error = self._show_error

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(60)
        self.refresh()

    # ==================================================================
    # construction
    # ==================================================================
    def _build_panels(self) -> None:
        self.project_panel = ProjectPanel(self.app)
        self.project_panel.newRequested.connect(self.new_project)
        self.project_panel.openRequested.connect(self.open_project)
        self.project_panel.openPathRequested.connect(self._open_path)
        self.project_panel.saveRequested.connect(self.save_project)
        self.project_panel.saveAsRequested.connect(self.save_project_as)

        self.brief_panel = BriefPanel(self.app)
        self.raaga_panel = RaagaPanel(self.app)
        for panel in (self.brief_panel, self.raaga_panel):
            panel.changed.connect(self.refresh)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(self.project_panel)
        left_layout.addWidget(self.brief_panel)
        left_layout.addWidget(self.raaga_panel)
        left_layout.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(330)
        left_scroll.setMaximumWidth(430)

        self.tune = TunePanel(self.app)
        self.lyrics = LyricsPanel(self.app)
        self.voice = VoicePanel(self.app)
        self.output = OutputPanel(self.app)
        for panel in (self.tune, self.lyrics, self.voice, self.output):
            panel.changed.connect(self.refresh)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tune, "Tune")
        self.tabs.addTab(self.lyrics, "Lyrics")
        self.tabs.addTab(self.voice, "Voice")
        self.tabs.addTab(self.output, "Output")

        self.arrangement = ArrangementPanel(self.app)
        self.arrangement.changed.connect(self.refresh)

        centre = QSplitter(Qt.Vertical)
        centre.addWidget(self.tabs)
        centre.addWidget(self.arrangement)
        centre.setSizes([420, 520])

        self.conversation = ConversationPanel(self.app)
        self.conversation.changed.connect(self.refresh)
        dock = QDockWidget("Conversation", self)
        dock.setWidget(self.conversation)
        dock.setFeatures(QDockWidget.DockWidgetMovable |
                         QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(330)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.conversation_dock = dock

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(centre)
        splitter.setSizes([360, 1000])
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Transport")
        bar.setMovable(False)
        self.addToolBar(bar)

        self.play_btn = QPushButton("Play")
        self.play_btn.setObjectName("primary")
        self.play_btn.clicked.connect(self._play)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.app.pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.app.stop)
        self.back_btn = QPushButton("-10s")
        self.back_btn.clicked.connect(lambda: self.app.playback.nudge(-10))
        self.fwd_btn = QPushButton("+10s")
        self.fwd_btn.clicked.connect(lambda: self.app.playback.nudge(10))

        self.position_label = QLabel("0:00.00 / 0:00.00")
        self.source_label = QLabel("-")
        self.source_label.setObjectName("hint")

        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 150)
        self.volume.setValue(100)
        self.volume.setFixedWidth(110)
        self.volume.valueChanged.connect(
            lambda v: self.app.playback.set_volume(v / 100.0))

        self.mic_btn = QPushButton("Mic")
        self.mic_btn.setCheckable(True)
        self.mic_btn.clicked.connect(self._toggle_mic)
        self.cancel_btn = QPushButton("Cancel work")
        self.cancel_btn.clicked.connect(
            lambda: self.app.jobs.cancel_all("cancelled by the creator"))

        for w in (self.play_btn, self.pause_btn, self.stop_btn, self.back_btn,
                  self.fwd_btn):
            bar.addWidget(w)
        bar.addSeparator()
        bar.addWidget(self.position_label)
        bar.addWidget(QLabel("  source:"))
        bar.addWidget(self.source_label)
        bar.addSeparator()
        bar.addWidget(QLabel("Volume"))
        bar.addWidget(self.volume)
        bar.addSeparator()
        bar.addWidget(self.mic_btn)
        bar.addWidget(self.cancel_btn)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        self._action(file_menu, "New project", QKeySequence.New, self.new_project)
        self._action(file_menu, "Open project...", QKeySequence.Open, self.open_project)
        self._action(file_menu, "Save", QKeySequence.Save, self.save_project)
        self._action(file_menu, "Save As...", QKeySequence.SaveAs, self.save_project_as)
        file_menu.addSeparator()
        self._action(file_menu, "Export full mix...", None,
                     lambda: self.output._export_audio("full"))
        self._action(file_menu, "Export vocal-only master...", None,
                     lambda: self.output._export_audio("vocal_master"))
        self._action(file_menu, "Archive project...", None, self.output._archive)
        file_menu.addSeparator()
        self._action(file_menu, "Quit", QKeySequence.Quit, self.close)

        edit_menu = menu.addMenu("&Edit")
        self.undo_action = self._action(edit_menu, "Undo", QKeySequence.Undo,
                                        self.app.undo_action)
        self.redo_action = self._action(edit_menu, "Redo", QKeySequence.Redo,
                                        self.app.redo_action)
        edit_menu.addSeparator()
        self._action(edit_menu, "Clear time selection", None,
                     lambda: (self.app.set_selection(None, None), self.refresh()))

        make_menu = menu.addMenu("&Compose")
        self._action(make_menu, "Generate tune", "Ctrl+T",
                     lambda: self.app.generate_tune())
        self._action(make_menu, "Tune variation", "Ctrl+Shift+T",
                     lambda: self.app.make_variation())
        self._action(make_menu, "Write lyrics", "Ctrl+L",
                     lambda: self.app.generate_lyrics())
        self._action(make_menu, "Render vocal preview", "Ctrl+R",
                     lambda: self.app.render_vocal("preview"))
        self._action(make_menu, "Studio vocal-only master", "Ctrl+Shift+R",
                     lambda: self.app.render_vocal("master"))
        self._action(make_menu, "Auto-arrange", "Ctrl+A",
                     lambda: self.app.auto_arrange())
        self._action(make_menu, "Render full mix", "Ctrl+M",
                     lambda: self.app.render("full"))

        voice_menu = menu.addMenu("&Voice control")
        self._action(voice_menu, "Start / stop listening", "Ctrl+Space",
                     self._toggle_mic)
        self._action(voice_menu, "Cancel current operation", "Esc",
                     lambda: self.app.jobs.cancel_all("cancelled by the creator"))

        help_menu = menu.addMenu("&Help")
        self._action(help_menu, "Voice command examples", None, self._show_help)
        self._action(help_menu, "Export diagnostics...", None, self._export_diagnostics)
        self._action(help_menu, "About", None, self._about)

    def _action(self, menu, text: str, shortcut, slot) -> QAction:  # noqa: ANN001
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut) if isinstance(shortcut, str)
                               else shortcut)
        action.triggered.connect(lambda: slot())
        menu.addAction(action)
        return action

    def _build_status(self) -> None:
        self.setStatusBar(QStatusBar())
        self.job_progress = QProgressBar()
        self.job_progress.setFixedWidth(190)
        self.job_progress.setRange(0, 100)
        self.job_progress.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.job_progress)
        self.statusBar().showMessage("Ready")

    # ==================================================================
    # project actions
    # ==================================================================
    def _confirm_discard(self) -> bool:
        if not self.app.dirty:
            return True
        answer = QMessageBox.question(
            self, "Unsaved changes",
            "Save the current project before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            self.save_project()
        return True

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.app.new_project("Untitled Song")
        self.refresh()

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Open project folder", str(self.app.store.projects_dir))
        if directory:
            self._open_path(directory)

    def _open_path(self, directory: str) -> None:
        try:
            self.app.open_project(Path(directory))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open project", str(exc))
            return
        self.refresh()

    def save_project(self) -> None:
        try:
            self.app.save()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save", str(exc))
        self.refresh()

    def save_project_as(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Save project as (choose an empty folder)",
            str(self.app.store.projects_dir))
        if directory:
            self.app.save_as(Path(directory))
            self.refresh()

    # ==================================================================
    # transport and timers
    # ==================================================================
    def _play(self) -> None:
        selection = self.app.selection
        if selection:
            self.app.play_range(selection[0], selection[1])
        elif self.app.playback.paused:
            self.app.resume()
        else:
            self.app.play_render()

    def _toggle_mic(self) -> None:
        if self.app.context.listening:
            self.app.stop_listening()
        else:
            self.app.start_listening()
        self.conversation.refresh()
        self.mic_btn.setChecked(self.app.context.listening)

    def _tick(self) -> None:
        self.app.pump()
        position = self.app.playback.position
        duration = self.app.playback.duration
        self.position_label.setText(f"{format_time(position)} / {format_time(duration)}")
        self.source_label.setText(self.app.playback.source_name or "-")
        self.arrangement.update_playhead()

        active = self.app.jobs.active_jobs()
        if active:
            job = active[0]
            self.job_progress.setFormat(f"{job.description} %p%")
            self.job_progress.setValue(int(job.progress * 100))
        else:
            self.job_progress.setFormat("idle")
            self.job_progress.setValue(0)
        self.mic_btn.setChecked(self.app.context.listening)
        self.conversation.refresh()

    # ==================================================================
    # refresh and messages
    # ==================================================================
    def refresh(self) -> None:
        self.project_panel.refresh()
        self.brief_panel.refresh()
        self.raaga_panel.refresh()
        self.tune.refresh()
        self.lyrics.refresh()
        self.voice.refresh()
        self.output.refresh()
        self.arrangement.refresh()
        self.undo_action.setText(f"Undo {self.app.undo.undo_label()}".strip())
        self.redo_action.setText(f"Redo {self.app.undo.redo_label()}".strip())
        self.undo_action.setEnabled(self.app.undo.can_undo)
        self.redo_action.setEnabled(self.app.undo.can_redo)
        title = self.app.project.title + (" *" if self.app.dirty else "")
        self.setWindowTitle(f"Raaga AI Music Composer - {title}")

    def _status_message(self, text: str) -> None:
        self.statusBar().showMessage(text, 12000)

    def _show_error(self, text: str) -> None:
        self.statusBar().showMessage(text, 20000)

    def _show_help(self) -> None:
        QMessageBox.information(self, "Voice commands", HELP_TEXT)

    def _about(self) -> None:
        QMessageBox.about(
            self, "Raaga AI Music Composer",
            "Raaga-aware interactive AI music composer\n"
            "A desktop music-director workstation.\n\n"
            f"Providers:\n{self.app.providers.summary()}")

    def _export_diagnostics(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export diagnostics", str(Path.home() / "raaga-diagnostics.zip"),
            "Zip (*.zip)")
        if path:
            out = self.app.export_diagnostics(Path(path))
            QMessageBox.information(self, "Diagnostics", f"Written to {out}")

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if not self._confirm_discard():
            event.ignore()
            return
        self._timer.stop()
        self.app.close()
        event.accept()


HELP_TEXT = """Speak or type instructions - both take the same path.

Playback
  "Play the first minute."
  "Play from the second minute to the third minute."
  "Play the end."   "Play the last 30 seconds."
  "Play from the chorus."   "Play this section again."
  "Start five seconds before this point."   "Go back 10 seconds."
  "Stop."   "Pause."   "Continue."

Arrangement
  "Add veena here."
  "Use saxophone for this interlude."
  "Bring strings after the chorus."
  "Take the drums out here."
  "Use only piano for the first 15 seconds."
  "Replace violin with veena."
  "Make this part lighter."
  "Give me another instrument that fits this feel."
  "I want this to feel lonely, late at night, but still warm."

Composition
  "Generate a tune."   "Give me a variation."
  "Use raaga Kalyani."   "Set the tempo to 96 bpm."
  "Write the lyrics."
  "Give me the song without instruments."
  "Mix the song."   "Lock the pallavi."   "Undo."

Interrupting
  Start speaking at any time - playback pauses and cancellable work stops
  before your sentence is finished. The newest instruction always wins.
"""


def run() -> int:
    import sys
    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setApplicationName("Raaga AI Music Composer")
    qt_app.setStyle("Fusion")
    qt_app.setStyleSheet(theme.STYLESHEET)
    controller = AppController()
    window = MainWindow(controller)
    window.show()
    return qt_app.exec()
