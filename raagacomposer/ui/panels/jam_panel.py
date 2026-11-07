"""The Jam Room tab (Krish, 7 Nov 2026): agents seated by role, the conductor's 1, 2, 3, go, and the team board as the room's door.

Each seat shows the agent, the role (conductor, lyricist, lead singer, lead / second instrument, bass, rhythm, drone), what they play
(an instrument icon, or a mic - blue for a male voice, pink for a female one) and whether they are connected: seated here, or
joined from the team board (green while they have been heard there in the last 15 minutes).

Count in: 20 seconds to get ready (Krish's rule), then the conductor's 1, 2, 3, GO - shown big on the screen and played as stick
clicks - and the band comes in.  With the room open on the board, the board hears the count-in too, so agents elsewhere listen at
the same moment, and any agent joins by posting  JAM JOIN <role> [instrument] [male|female].
"""
from __future__ import annotations

import logging
import os
import threading
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...studio import jam as J

log = logging.getLogger(__name__)

COUNTDOWN_S = 20
POLL_MS = 8000
FRESH_S = 15 * 60
JOIN_HELP = ("Agents join from the team board: JAM JOIN <role> [instrument] [male|female]  - roles: conductor, lyricist, singer, "
             "lead, second, bass, rhythm, drone.  JAM LEAVE to step out.")


def _label(key: str) -> str:
    return key.replace("_", " ").title()


class JamSeatCard(QFrame):
    changed = Signal()

    def __init__(self, seat: J.JamSeat, parent=None) -> None:
        super().__init__(parent)
        self.seat = seat
        self.setObjectName("jamseat")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.icon.font()
        f.setPointSize(28)
        self.icon.setFont(f)
        self.name = QLabel()
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.link = QLabel()
        self.link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.role = QComboBox()
        for key in J.ROLES:
            self.role.addItem(J.ROLE_NAMES[key], key)
        self.instrument = QComboBox()
        self.voice = QComboBox()
        self.voice.addItems(["female", "male"])
        self.playing = QCheckBox("In the jam")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        for w in (self.icon, self.name, self.link, self.role, self.instrument, self.voice, self.playing):
            lay.addWidget(w)
        self.role.activated.connect(self._role_changed)
        self.instrument.activated.connect(self._edit)
        self.voice.activated.connect(self._edit)
        self.playing.toggled.connect(self._edit)
        self.show_seat()
        self.set_lit(False)

    def show_seat(self) -> None:
        s = self.seat
        self.icon.setText(s.icon)
        self.name.setText(f"<b>{s.agent}</b>")
        if s.joined == "board":
            fresh = s.seen and time.time() - s.seen < FRESH_S
            self.link.setText("<span style='color:#22c55e'>● on the board</span>" if fresh
                              else "<span style='color:#9ca3af'>○ board (quiet)</span>")
        else:
            self.link.setText("<span style='color:#9ca3af'>seated here</span>")
        self.role.setCurrentIndex(max(0, self.role.findData(s.role)))
        allowed = J.CHOICES.get(s.role, ())
        self.instrument.blockSignals(True)
        self.instrument.clear()
        for key in allowed:
            self.instrument.addItem(_label(key), key)
        self.instrument.setCurrentIndex(max(0, self.instrument.findData(s.instrument)))
        self.instrument.blockSignals(False)
        self.instrument.setVisible(bool(allowed))
        self.voice.setVisible(s.role in (J.SINGER, J.CONDUCTOR))
        self.voice.setCurrentText(s.voice if s.voice in ("female", "male") else "female")
        self.playing.blockSignals(True)
        self.playing.setChecked(s.playing)
        self.playing.blockSignals(False)

    def _role_changed(self, *_args) -> None:
        self.seat.role = str(self.role.currentData())
        J.fix_seat(self.seat)
        self._finish()

    def _edit(self, *_args) -> None:
        if self.instrument.isVisible() and self.instrument.currentData():
            self.seat.instrument = str(self.instrument.currentData())
        if self.voice.isVisible():
            self.seat.voice = self.voice.currentText()
        self.seat.playing = self.playing.isChecked()
        self._finish()

    def _finish(self) -> None:
        J.fix_seat(self.seat)
        self.show_seat()
        self.set_lit(False)
        self.changed.emit()

    def set_lit(self, lit: bool) -> None:
        s = self.seat
        border = s.mic_colour if s.role in (J.SINGER, J.CONDUCTOR) else ("#f59e0b" if s.role == J.LYRICIST else "#6b7280")
        back = "#3b3416" if (lit and s.playing) else ("#1f2430" if s.playing else "#14161c")
        self.setStyleSheet(f"QFrame#jamseat {{ border: 3px solid {border}; border-radius: 10px; background: {back}; }}")


class JamPanel(QWidget):
    changed = Signal()
    jam_ready = Signal(object)
    board_seen = Signal(object)          # messages read off the board (from a worker thread)

    def __init__(self, app, parent=None, board: bool | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.seats = J.default_room()
        self.cards: list[JamSeatCard] = []
        self.grid = QGridLayout()
        holder = QWidget()
        holder.setLayout(self.grid)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.go_btn = QPushButton(f"Get ready ({COUNTDOWN_S} s), then 1, 2, 3, go")
        self.go_btn.setObjectName("primary")
        self.go_btn.clicked.connect(self.count_in)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.board_box = QCheckBox("Open the room on the team board")
        # Opt-in (review): the room only talks to the board when the director ticks this, so a launch never posts by itself.
        on = board if board is not None else ("PYTEST_CURRENT_TEST" not in os.environ and os.environ.get("RAAGA_JAM_BOARD") == "1")
        self.board_box.setChecked(on)
        self.board_box.toggled.connect(self._board_toggled)
        top = QHBoxLayout()
        top.addWidget(self.go_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.board_box)
        top.addStretch(1)

        self.big = QLabel("")
        self.big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.big.font()
        f.setPointSize(30)
        f.setBold(True)
        self.big.setFont(f)
        self.big.setStyleSheet("color:#fbbf24")
        self.status = QLabel("Krish conducts. Seat the band, then count it in.")
        self.status.setWordWrap(True)
        self.ready = QLabel()
        self.ready.setWordWrap(True)

        self.mood = QLineEdit("a joyful jam with friends, bright and bouncy")
        self.mood.setPlaceholderText("The mood, for the lyricist")
        self.write_btn = QPushButton("Lyricist: write two lines")
        self.write_btn.clicked.connect(self.write_lines)
        self.lyrics = QLabel("")
        self.lyrics.setWordWrap(True)
        lyr = QHBoxLayout()
        lyr.addWidget(QLabel("Mood:"))
        lyr.addWidget(self.mood, 1)
        lyr.addWidget(self.write_btn)

        self.seat_name = QLineEdit()
        self.seat_name.setPlaceholderText("Seat an agent here by name")
        self.seat_name.returnPressed.connect(self.add_seat)
        add_btn = QPushButton("Add seat")
        add_btn.clicked.connect(self.add_seat)
        add = QHBoxLayout()
        add.addWidget(self.seat_name, 1)
        add.addWidget(add_btn)
        help_ = QLabel(JOIN_HELP)
        help_.setWordWrap(True)
        help_.setStyleSheet("color:#9ca3af")

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.big)
        lay.addWidget(self.status)
        lay.addWidget(self.ready)
        lay.addLayout(lyr)
        lay.addWidget(self.lyrics)
        lay.addWidget(scroll, 1)
        lay.addLayout(add)
        lay.addWidget(help_)
        self.setMinimumWidth(480)
        self.setMinimumHeight(300)

        self._rendered = None
        self._busy = False                           # counting down or rendering: the count-in button waits
        self._run = 0                                # which count-in a finished render belongs to
        self._left = 0
        self._opened_at = time.time() - 600          # joins posted in the last ten minutes count
        self._announced = False
        self._reading = False
        self._applied: set[str] = set()              # board messages already acted on: each one acts once
        self._lit = False
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._count)
        self._glow = QTimer(self)
        self._glow.setSingleShot(True)
        self._glow.timeout.connect(self._done_playing)
        self._beats: list[QTimer] = []
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._read_board)
        self.jam_ready.connect(self._play)
        self.board_seen.connect(self._board_messages)
        self._rebuild()
        if on:
            self._board_toggled(True)

    # -- seats -------------------------------------------------------------
    def _rebuild(self) -> None:
        for card in self.cards:
            card.setParent(None)
            card.deleteLater()
        self.cards = [JamSeatCard(s) for s in self.seats]
        for k, card in enumerate(self.cards):
            self.grid.addWidget(card, k // 6, k % 6)
            card.changed.connect(self._seats_changed)
            card.set_lit(self._lit)
        self._seats_changed()

    def _seats_changed(self) -> None:
        ok, notes = J.check(self.seats)
        head = "<b style='color:#22c55e'>Ready to jam.</b> " if ok else "<b style='color:#f87171'>Not ready:</b> "
        self.ready.setText(head + " ".join(notes))
        self.go_btn.setEnabled(ok and not self._busy)
        self.changed.emit()

    def add_seat(self) -> None:
        name = " ".join(self.seat_name.text().split())
        if not name:
            return
        if any(s.agent.lower() == name.lower() for s in self.seats):
            self.status.setText(f"{name} already has a seat.")
            return
        self.seats.append(J.fix_seat(J.JamSeat(name, J.RHYTHM, "tabla")))
        self.seat_name.clear()
        self._rebuild()
        self.status.setText(f"{name} sits in on tabla; change the role on the card.")

    # -- lyricist ----------------------------------------------------------
    def write_lines(self) -> None:
        mood = self.mood.text().strip()
        self.write_btn.setEnabled(False)
        who = next((s.agent for s in self.seats if s.role == J.LYRICIST and s.playing), "The lyricist")
        self.app.jobs.submit("jam.lyrics", "jam-lyrics", lambda ctx: J.lyricist_lines(mood),
                             on_done=lambda lines: self._lines(who, lines), on_error=self._lines_failed,
                             description="The lyricist writes two lines")

    def _lines(self, who: str, lines) -> None:
        self.write_btn.setEnabled(True)
        self.lyrics.setText(f"<i>{who} wrote:</i><br>" + "<br>".join(str(x) for x in lines))
        self._post(f"JAM lyricist {who}: " + " / ".join(str(x) for x in lines))

    def _lines_failed(self, exc) -> None:
        self.write_btn.setEnabled(True)
        self.lyrics.setText(f"The lyricist could not write: {exc}")

    # -- the board ---------------------------------------------------------
    def _board_toggled(self, on: bool) -> None:
        if on:
            self._poll.start(POLL_MS)
            self._read_board()
            if not self._announced:
                self._announced = True
                conductor = next((s.agent for s in self.seats if s.role == J.CONDUCTOR and s.playing), "the room")
                self._post(f"JAM ROOM OPEN on the raga app. {conductor} conducts. " + JOIN_HELP)
        else:
            self._poll.stop()

    def _post(self, text: str) -> None:
        if not self.board_box.isChecked():
            return
        threading.Thread(target=self._post_quietly, args=(text,), daemon=True).start()

    @staticmethod
    def _post_quietly(text: str) -> None:
        try:
            J.board_post(text)
        except (OSError, ValueError) as exc:             # the board being down never stops the jam
            log.info("jam: board post failed: %s", exc)

    def _read_board(self) -> None:
        if self._reading:
            return
        self._reading = True

        def work() -> None:
            got = None
            try:
                got = J.board_read()
            except Exception as exc:  # noqa: BLE001 - any failure must still emit, or polling stops for good (review)
                log.info("jam: board read failed: %s", exc)
            finally:
                self.board_seen.emit(got)
        threading.Thread(target=work, daemon=True).start()

    def _board_messages(self, messages) -> None:
        self._reading = False
        if messages is None:
            self.board_box.setText("Open the room on the team board (board not reachable)")
            return
        self.board_box.setText("Open the room on the team board")
        before = len(self.seats)
        changes = J.apply_board(self.seats, messages, self._opened_at, self._applied)
        if len(self.seats) != before:
            self._rebuild()
        else:
            for card in self.cards:
                card.show_seat()
                card.set_lit(self._lit)                  # a poll never puts the band's lights out mid-jam
            self._seats_changed()
        if changes:
            self.status.setText("From the board: " + "; ".join(changes) + ".")

    # -- count in, render, play -------------------------------------------
    def count_in(self) -> None:
        ok, notes = J.check(self.seats)
        if not ok:
            self.status.setText("Not ready: " + " ".join(notes))
            return
        self._busy = True
        self.go_btn.setEnabled(False)
        self._rendered = None
        self._run += 1
        run = self._run
        seats = [J.JamSeat(**s.to_dict()) for s in self.seats]          # a snapshot: no seat changes mid-render
        sr = self.app.sample_rate
        self.app.jobs.submit("jam.render", "jam", lambda ctx: J.render(seats, sr=sr),
                             on_done=lambda audio: self._rendered_ok(audio, run),
                             on_error=lambda exc: self._render_failed(exc, run), description="Render the jam")
        self._left = COUNTDOWN_S
        self._post(f"JAM: get ready - count-in in {COUNTDOWN_S} s. Lineup: {J.lineup(self.seats)}.")
        self._count()
        self._tick.start(1000)

    def _rendered_ok(self, audio, run: int = 0) -> None:
        if run != self._run or not self._busy:     # stopped, or a newer count-in: this render is not wanted
            return
        self._rendered = audio
        if self._left <= 0 and not self._tick.isActive():
            self.jam_ready.emit(audio)

    def _render_failed(self, exc, run: int = 0) -> None:
        if run != self._run:
            return
        self._busy = False
        self._tick.stop()
        self.big.setText("")
        self._seats_changed()
        self.status.setText(f"The jam could not be rendered: {exc}")

    def _count(self) -> None:
        if self._left > 0:
            self.big.setText(f"Get ready… {self._left}")
            self._left -= 1
            return
        self._tick.stop()
        if self._rendered is not None:
            self.jam_ready.emit(self._rendered)
        else:
            self.big.setText("Tuning up…")

    def _play(self, audio) -> None:
        if not self._busy:
            return
        self.app._cache_render("jam", audio)          # _busy stays on while it plays: Stop first, then a new count-in
        self.app.play_render("jam")
        conductor = next((s.agent for s in self.seats if s.role == J.CONDUCTOR and s.playing), "")
        beat_ms = int(1000 * 60.0 / J.G.BPM)
        self._clear_beats()
        for i, word in enumerate(J.COUNT):
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda w=word: self.big.setText(w.upper() + "!" if w == "go" else w))
            t.start(i * beat_ms)
            self._beats.append(t)
        self._post(f"JAM: {conductor or 'the room'} counts 1, 2, 3, GO! ({J.G.BPM} bpm, Shanmukhapriya in D).")
        self._light(True)
        self._glow.start(int(1000 * len(audio) / self.app.sample_rate))
        self.status.setText("Jamming: " + J.lineup(self.seats))
        self._seats_changed()

    def _clear_beats(self) -> None:
        for t in self._beats:
            t.stop()
            t.deleteLater()
        self._beats = []

    def _light(self, lit: bool) -> None:
        self._lit = lit
        for card in self.cards:
            card.set_lit(lit)

    def _done_playing(self) -> None:
        self._busy = False
        self._light(False)
        self.big.setText("")
        self._seats_changed()

    def stop(self) -> None:
        was = self._busy or self._glow.isActive()
        self._busy = False
        self._run += 1
        self._tick.stop()
        self._glow.stop()
        self._clear_beats()
        self.app.jobs.cancel_target("jam", "stopped")
        self._left = 0
        self._rendered = None
        self.app.stop()
        self._light(False)
        self.big.setText("")
        self.status.setText("Stopped.")
        self._seats_changed()
        if was:
            self._post("JAM: stopped.")
