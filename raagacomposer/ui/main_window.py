"""Main desktop window (spec section 14; workspaces per spec section 4).

Two top-level workspaces share this one window: MAIN, the composition
experience - project header, creative brief and raaga down the left; tune,
lyrics, voice and output as tabs in the centre above the arrangement
timeline - and LEARN, the agent's own screen (see ``learn_workspace.py``).
A toolbar toggle and a View menu switch between them; the conversation dock,
transport and status bar are shared by both.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDockWidget,
                               QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSplitter,
                               QStackedWidget, QStatusBar, QTabWidget, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from ..app import AppController
from ..core.logging_setup import get_logger
from ..music.theory import format_time
from . import theme
from .learn_workspace import LearnWorkspace, provider_summary_line
from .panels.agent_panel import AgentPanel
from .panels.training_panel import TrainingPanel
from .panels.arrangement_panel import ArrangementPanel
from .panels.brief_panel import BriefPanel
from .panels.conversation_panel import ConversationPanel
from .panels.lyrics_panel import LyricsPanel
from .panels.output_panel import OutputPanel
from .panels.raaga_panel import RaagaPanel
from .panels.tune_panel import TunePanel
from .panels.voice_panel import VoicePanel
from .settings_dialog import SettingsDialog

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

        # Provider status (spec 41) is read on its own, slower cadence - the
        # status bar label and it never need to be fresher than 30s, and some
        # backends' ``.status()`` probes real infrastructure (Ollama's HTTP
        # check) that need not run on every 60ms tick.
        self._provider_timer = QTimer(self)
        self._provider_timer.timeout.connect(self._refresh_provider_status)
        self._provider_timer.start(30000)
        self._refresh_provider_status()

        self.refresh()
        # Two top-level workspaces (spec section 4): restore whichever one
        # was open last, MAIN by default.
        self.set_workspace(self.app.settings.extra.get("workspace", "MAIN"))

    # ==================================================================
    # construction
    # ==================================================================
    def _build_panels(self) -> None:

        self.brief_panel = BriefPanel(self.app)
        self.raaga_panel = RaagaPanel(self.app)
        for panel in (self.brief_panel, self.raaga_panel):
            panel.changed.connect(self.refresh)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
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
        # The learning agent and training panels are built here (MAIN and
        # LEARN share one AppController and one set of widget instances) but
        # are no longer added to this tab bar - spec section 4 makes LEARN a
        # separate top-level workspace, not a tab beside Tune and Lyrics.
        self.agent_panel = AgentPanel(self.app)
        self.training_panel = TrainingPanel(self.app)
        for panel in (self.tune, self.lyrics, self.voice, self.output,
                      self.agent_panel, self.training_panel):
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
        self.main_page = splitter

        # LEARN: a separate full workspace (spec 4.2), built by re-homing the
        # widgets already inside agent_panel and training_panel.
        self.learn_workspace = LearnWorkspace(
            self.app, self.agent_panel, self.training_panel,
            open_settings=self._open_settings)

        self.workspaces = QStackedWidget()
        self.workspaces.addWidget(self.main_page)   # index 0: MAIN
        self.workspaces.addWidget(self.learn_workspace)  # index 1: LEARN
        self.setCentralWidget(self.workspaces)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Transport")
        bar.setMovable(False)
        self.addToolBar(bar)

        # Workspace switch (spec section 4): two mutually exclusive toggle
        # buttons at the left end of the transport toolbar, ahead of every
        # transport control.
        self.main_ws_btn = QToolButton()
        self.main_ws_btn.setText("MAIN")
        self.main_ws_btn.setCheckable(True)
        self.learn_ws_btn = QToolButton()
        self.learn_ws_btn.setText("LEARN")
        self.learn_ws_btn.setCheckable(True)
        self._workspace_group = QButtonGroup(self)
        self._workspace_group.setExclusive(True)
        self._workspace_group.addButton(self.main_ws_btn)
        self._workspace_group.addButton(self.learn_ws_btn)
        self.main_ws_btn.clicked.connect(lambda: self.set_workspace("MAIN"))
        self.learn_ws_btn.clicked.connect(lambda: self.set_workspace("LEARN"))
        bar.addWidget(self.main_ws_btn)
        bar.addWidget(self.learn_ws_btn)
        bar.addSeparator()

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
        # Where every other application keeps it, rather than a dropdown in
        # a panel taking 179px off the top of the composing column.
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
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
        edit_menu.addSeparator()
        self._action(edit_menu, "Settings...", "Ctrl+,", self._open_settings)

        view_menu = menu.addMenu("&View")
        self.main_ws_action = QAction("Main workspace", self)
        self.main_ws_action.setCheckable(True)
        self.main_ws_action.setShortcut(QKeySequence("Ctrl+1"))
        self.main_ws_action.triggered.connect(lambda: self.set_workspace("MAIN"))
        self.learn_ws_action = QAction("Learn workspace", self)
        self.learn_ws_action.setCheckable(True)
        self.learn_ws_action.setShortcut(QKeySequence("Ctrl+2"))
        self.learn_ws_action.triggered.connect(lambda: self.set_workspace("LEARN"))
        self._workspace_actions = QActionGroup(self)
        self._workspace_actions.setExclusive(True)
        self._workspace_actions.addAction(self.main_ws_action)
        self._workspace_actions.addAction(self.learn_ws_action)
        view_menu.addAction(self.main_ws_action)
        view_menu.addAction(self.learn_ws_action)

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

        learn_menu = menu.addMenu("&Learning")
        self._action(learn_menu, "Open the Learn workspace", None,
                     lambda: self.set_workspace("LEARN"))
        learn_menu.addSeparator()
        self._action(learn_menu, "Start / pause learning", "Ctrl+Shift+L",
                     self.agent_panel._toggle_learning)
        self._action(learn_menu, "Study one lesson now", None,
                     self.agent_panel._step)
        self._action(learn_menu, "Stop learning", None, self.app.stop_learning)
        learn_menu.addSeparator()
        self._action(learn_menu, "Mark the current tune", None,
                     self.agent_panel._critique)
        self._action(learn_menu, "Choose my learning folder...", None,
                     self.agent_panel._choose_corpus)
        self._action(learn_menu, "What the agent knows", None,
                     self._show_knowledge)

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

        # Provider status (spec 41): never shows a secret, only state words.
        self.provider_status_label = QLabel("-")
        self.provider_status_label.setObjectName("hint")
        self.provider_settings_btn = QPushButton("Settings")
        self.provider_settings_btn.setFlat(True)
        self.provider_settings_btn.clicked.connect(self._open_settings)
        self.project_status_label = QLabel("")
        self.project_status_label.setObjectName("hint")
        # A permanent status widget that sizes to its text drags the whole
        # window's minimum width along with it - this one cost 226px before
        # it was made to shrink, which is the opposite of the point of
        # moving the project header off the left column.  Ignored means the
        # status bar may squeeze it to nothing when space is short; the full
        # path is in the tooltip either way.
        self.project_status_label.setSizePolicy(QSizePolicy.Ignored,
                                                QSizePolicy.Preferred)
        self.project_status_label.setMaximumWidth(320)
        self.statusBar().addPermanentWidget(self.project_status_label)
        self.statusBar().addPermanentWidget(self.provider_status_label)
        self.statusBar().addPermanentWidget(self.provider_settings_btn)

        self.job_progress = QProgressBar()
        self.job_progress.setFixedWidth(190)
        self.job_progress.setRange(0, 100)
        self.job_progress.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.job_progress)
        self.statusBar().showMessage("Ready")

    # ==================================================================
    # workspaces (spec section 4) and settings (spec 41, 42)
    # ==================================================================
    def set_workspace(self, name: str) -> None:
        """Switch between the MAIN and LEARN top-level workspaces."""
        name = "LEARN" if name == "LEARN" else "MAIN"
        self.workspaces.setCurrentIndex(1 if name == "LEARN" else 0)
        self.main_ws_btn.setChecked(name == "MAIN")
        self.learn_ws_btn.setChecked(name == "LEARN")
        self.main_ws_action.setChecked(name == "MAIN")
        self.learn_ws_action.setChecked(name == "LEARN")
        self.app.settings.extra["workspace"] = name
        self.app.settings.save()
        if name == "LEARN":
            self.learn_workspace.refresh()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.app, self)
        dialog.exec()
        self._refresh_provider_status()

    def _refresh_provider_status(self) -> None:
        text = provider_summary_line(self.app.provider_statuses())
        self.provider_status_label.setText(text)
        if hasattr(self, "learn_workspace"):
            self.learn_workspace.set_provider_line(text)

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
            self, "Save song as (choose or name a folder)",
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
        self.brief_panel.refresh()
        self.raaga_panel.refresh()
        self.tune.refresh()
        self.lyrics.refresh()
        self.voice.refresh()
        self.output.refresh()
        self.agent_panel.refresh()
        self.learn_workspace.refresh()
        self.arrangement.refresh()
        self.undo_action.setText(f"Undo {self.app.undo.undo_label()}".strip())
        self.redo_action.setText(f"Redo {self.app.undo.redo_label()}".strip())
        self.undo_action.setEnabled(self.app.undo.can_undo)
        self.redo_action.setEnabled(self.app.undo.can_redo)
        # The song's name leads, the way a document's does: "Kaadhal - Raaga
        # AI Music Composer", with the unsaved mark beside the name it
        # belongs to.
        title = self.app.project.title + (" *" if self.app.dirty else "")
        self.setWindowTitle(f"{title} - Raaga AI Music Composer")
        project = self.app.project
        # The folder's name is the useful part on screen; the whole path is
        # a tooltip away for anyone who needs it.
        directory = self.app.project_dir
        self.project_status_label.setText(
            f"{project.current_stage.value} - "
            f"{project.brief.duration_target:.0f}s - "
            f"{directory.name if directory else 'not saved yet'}")
        self.project_status_label.setToolTip(
            str(directory) if directory else "This song has not been saved yet")

    def _fill_recent_menu(self) -> None:
        """Rebuilt each time it opens, so it is never stale."""
        self.recent_menu.clear()
        try:
            entries = self.app.recent_projects()
        except Exception as exc:  # noqa: BLE001 - a menu is never worth a crash
            log.warning("could not list recent projects: %s", exc)
            entries = []
        if not entries:
            empty = self.recent_menu.addAction("Nothing opened yet")
            empty.setEnabled(False)
            return
        for entry in entries[:10]:
            path = str(entry.get("path", ""))
            label = str(entry.get("title", "")) or Path(path).name
            action = self.recent_menu.addAction(f"{label}    {path}")
            action.triggered.connect(lambda _=False, p=path: self._open_path(p))

    def _status_message(self, text: str) -> None:
        self.statusBar().showMessage(text, 12000)

    def _show_error(self, text: str) -> None:
        self.statusBar().showMessage(text, 20000)

    def _show_knowledge(self) -> None:
        self.set_workspace("LEARN")
        self.learn_workspace.show_area("Knowledge")
        self.agent_panel.refresh()

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
        self._provider_timer.stop()
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
