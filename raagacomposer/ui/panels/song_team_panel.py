"""Song Team panel: the producer's desk for the simple path.

Type a mood; the mood listener says what it heard; the lyricist and the
tunesmith work at the same time; the tala keeper fits the beat; the rough
picture plays.  The producer can overrule the raaga or the tala and make it
again.  What is not built yet is listed on the panel, not hidden.

Design note: ``docs/SONG_TEAM.md``.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...music import tala as tala_module
from ...studio.team import LANGUAGES, NOT_DONE_YET

AUTO_RAAGA = "Let the tunesmith choose"
AUTO_TALA = "Let the tala keeper choose"


def _box(title: str) -> tuple:
    group = QGroupBox(title)
    text = QTextEdit()
    text.setReadOnly(True)
    text.setMinimumHeight(70)
    lay = QVBoxLayout(group)
    lay.setContentsMargins(6, 6, 6, 6)
    lay.addWidget(text)
    return group, text


class SongTeamPanel(QWidget):
    changed = Signal()
    #: The controller calls back from ``pump``; a signal keeps it on the
    #: UI thread whatever the caller (the same rule as MainWindow's).
    draft_arrived = Signal(object)

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        self.mood_edit = QLineEdit()
        self.mood_edit.setPlaceholderText(
            "Describe the mood, the way you would to a producer - "
            "e.g. lonely, late at night, but still warm")
        self.mood_edit.returnPressed.connect(self._make)
        self.language_box = QComboBox()
        self.language_box.addItems(list(LANGUAGES))
        self.raaga_box = QComboBox()
        self.raaga_box.addItem(AUTO_RAAGA, "")
        for name in sorted(r.name for r in self.app.raagas.all()):
            self.raaga_box.addItem(name, name)
        self.tala_box = QComboBox()
        self.tala_box.addItem(AUTO_TALA, "")
        for name in tala_module.names():
            self.tala_box.addItem(name, name)

        self.make_btn = QPushButton("Make the rough song")
        self.make_btn.setObjectName("primary")
        self.make_btn.clicked.connect(self._make)
        self.play_btn = QPushButton("Play rough song")
        self.play_btn.clicked.connect(self.app.play_song_draft)
        self.play_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.app.stop)
        self.status_label = QLabel("Type a mood and press Make.")
        self.status_label.setWordWrap(True)

        top = QGridLayout()
        top.addWidget(QLabel("Mood"), 0, 0)
        top.addWidget(self.mood_edit, 0, 1, 1, 5)
        top.addWidget(QLabel("Language"), 1, 0)
        top.addWidget(self.language_box, 1, 1)
        top.addWidget(QLabel("Raaga"), 1, 2)
        top.addWidget(self.raaga_box, 1, 3)
        top.addWidget(QLabel("Tala"), 1, 4)
        top.addWidget(self.tala_box, 1, 5)
        top.setColumnStretch(1, 1)
        top.setColumnStretch(3, 1)
        top.setColumnStretch(5, 1)

        buttons = QHBoxLayout()
        for w in (self.make_btn, self.play_btn, self.stop_btn):
            buttons.addWidget(w)
        buttons.addStretch(1)

        # What the mood listener heard is one sentence; it gets a line, not
        # a box, so the three agents' cards keep the height.
        self.mood_text = QLabel("1. Mood listener: waiting for a mood.")
        self.mood_text.setWordWrap(True)
        lyric_box, self.lyrics_text = _box("2a. Lyricist")
        tune_box, self.tune_text = _box("2b. Tunesmith - raaga and phrase")
        beat_box, self.beat_text = _box("3. Tala keeper - beat")
        cards = QHBoxLayout()
        cards.addWidget(lyric_box, 1)
        cards.addWidget(tune_box, 1)
        cards.addWidget(beat_box, 1)

        log_box, self.log_text = _box("Team log - who handed what to whom")
        todo_box, self.todo_text = _box("Not done yet")
        self.todo_text.setPlainText("\n".join(f"- {x}" for x in NOT_DONE_YET))
        bottom = QHBoxLayout()
        bottom.addWidget(log_box, 3)
        bottom.addWidget(todo_box, 2)

        # The tab shares its height with the arrangement below it, so the
        # content scrolls rather than squashing the cards to slivers.
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(4, 4, 4, 4)
        inner.addLayout(top)
        inner.addLayout(buttons)
        inner.addWidget(self.status_label)
        inner.addWidget(self.mood_text)
        inner.addLayout(cards, 3)
        inner.addLayout(bottom, 2)
        content.setMinimumHeight(560)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setMinimumWidth(480)
        self.setMinimumHeight(260)

        self.draft_arrived.connect(self.show_draft)
        self.app.on_song_draft = self.draft_arrived.emit

    # -- actions -----------------------------------------------------------
    def _make(self) -> None:
        if not self.make_btn.isEnabled():               # Enter in the mood box while a run is working
            return
        mood = self.mood_edit.text().strip()
        if not mood:
            self.status_label.setText("Describe the mood in a few words first.")
            return
        started = self.app.make_song_draft(
            mood, self.language_box.currentText(),
            raaga=str(self.raaga_box.currentData() or ""),
            tala=str(self.tala_box.currentData() or ""))
        if started:
            self.status_label.setText("The song team is working...")
            self.make_btn.setEnabled(False)

    # -- display -----------------------------------------------------------
    def show_draft(self, draft) -> None:
        self.make_btn.setEnabled(True)
        if draft is None:
            return
        mood, lyrics, tune, beat = draft.mood, draft.lyrics, draft.tune, draft.beat
        self.mood_text.setText(f"1. Mood listener: {mood.summary()}")

        sung = []
        if lyrics.fitted is not None:
            sung = [f"  {l.text}   [{'-'.join(l.syllables)}]"
                    for l in lyrics.fitted.lines]
        revised = (f"\nRewritten to fit the tune: line "
                   f"{', '.join(str(i + 1) for i in lyrics.revised)}"
                   if lyrics.revised else "")
        self.lyrics_text.setPlainText(
            f"Written by: {lyrics.written_by}\n"
            + "\n".join(lyrics.lines)
            + revised
            + ("\n\nAs sung, phrase by phrase:\n" + "\n".join(sung)
               if sung else ""))

        alts = "\n".join(f"  {name}" for name, _ in tune.alternatives
                         if name != tune.raaga)
        self.tune_text.setPlainText(
            f"{tune.raaga} (chosen by the {tune.chosen_by})\n{tune.reason}\n\n"
            f"Arohanam:   {' '.join(tune.arohanam)}\n"
            f"Avarohanam: {' '.join(tune.avarohanam)}\n"
            f"Alapana:    {' '.join(n.swara for n in tune.alapana)}"
            f"  ({tune.alapana_seconds:.1f}s, free time)\n"
            + "\n".join(f"Line {i + 1}:     {line}"
                        for i, line in enumerate(tune.phrase_lines))
            + (f"\n\nOther raagas that fit:\n{alts}" if alts else ""))

        self.beat_text.setPlainText(
            f"{beat.description} (chosen by the {beat.chosen_by})\n"
            f"{beat.reason}\n{beat.tempo_bpm} bpm, {beat.density}, "
            f"{len(beat.beat.notes)} strokes"
            + ("\nThe tune was re-set to this cycle." if beat.recast_tune else ""))

        self.log_text.setPlainText("\n".join(e.line() for e in draft.events))
        rough = draft.rough
        if rough is not None:
            self.status_label.setText(
                f"Rough song: {rough.seconds:.0f}s - " + "; ".join(rough.layers))
            self.play_btn.setEnabled(True)
        self.changed.emit()
