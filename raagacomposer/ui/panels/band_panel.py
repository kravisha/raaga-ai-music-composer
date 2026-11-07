"""The Agentic Band's stage: one seat per agent (Krish, 7 Nov 2026).

Each seat shows the agent, what they play (an instrument, or a mic - blue for a male voice, pink for a female one), and whether
they are playing. The director changes a seat's instrument or sits an agent out, then counts the band in: a 20-second countdown
first (Krish's rule: time to get ready to listen), then the band plays and the playing seats light up.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...studio import band as B

COUNTDOWN_S = 20
ROLE_CHOICES = [("Vocal (mic)", B.VOCAL)] + [(k.replace("_", " ").title(), k) for k in B.PLAYABLE]


class SeatCard(QFrame):
    changed = Signal()

    def __init__(self, seat: B.Seat, parent=None) -> None:
        super().__init__(parent)
        self.seat = seat
        self.setObjectName("seat")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = self.icon.font()
        f.setPointSize(30)
        self.icon.setFont(f)
        self.name = QLabel()
        self.name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.role = QComboBox()
        for label, key in ROLE_CHOICES:
            self.role.addItem(label, key)
        self.voice = QComboBox()
        self.voice.addItems(["female", "male"])
        self.playing = QCheckBox("Playing")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        for w in (self.icon, self.name, self.role, self.voice, self.playing):
            lay.addWidget(w)
        self.role.activated.connect(self._edit)
        self.voice.activated.connect(self._edit)
        self.playing.toggled.connect(self._edit)
        self.show_seat()
        self.set_lit(False)

    def show_seat(self) -> None:
        s = self.seat
        self.icon.setText(s.icon)
        title = f"<br><small>{s.title}</small>" if s.title else ""
        self.name.setText(f"<b>{s.agent}</b>{title}")
        key = B.VOCAL if s.role == B.VOCAL else s.instrument
        i = self.role.findData(key)
        self.role.setCurrentIndex(max(0, i))
        self.voice.setVisible(s.role == B.VOCAL)
        self.voice.setCurrentText(s.voice if s.voice in ("female", "male") else "female")
        self.playing.blockSignals(True)
        self.playing.setChecked(s.playing)
        self.playing.blockSignals(False)

    def _edit(self, *_args) -> None:
        key = str(self.role.currentData())
        if key == B.VOCAL:
            self.seat.role, self.seat.instrument = B.VOCAL, ""
            self.seat.voice = self.voice.currentText()
        else:
            self.seat.role, self.seat.instrument = B.INSTRUMENT, key
        self.seat.playing = self.playing.isChecked()
        self.show_seat()
        self.set_lit(False)
        self.changed.emit()

    def set_lit(self, lit: bool) -> None:
        s = self.seat
        border = s.mic_colour if s.role == B.VOCAL else "#6b7280"
        back = "#3b3416" if (lit and s.playing) else ("#1f2430" if s.playing else "#14161c")
        self.setStyleSheet(f"QFrame#seat {{ border: 3px solid {border}; border-radius: 10px; background: {back}; }}")


class BandPanel(QWidget):
    changed = Signal()
    band_ready = Signal(object)

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self.seats = B.default_band()
        self.cards = [SeatCard(s) for s in self.seats]
        grid = QGridLayout()
        for k, card in enumerate(self.cards):
            grid.addWidget(card, k // 6, k % 6)
            card.changed.connect(self._seats_changed)
        holder = QWidget()
        holder.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.count_btn = QPushButton(f"Count me in ({COUNTDOWN_S} s), then play")
        self.count_btn.setObjectName("primary")
        self.count_btn.clicked.connect(self.count_in)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.status = QLabel("Krish directs; Kali produces. Change a seat's instrument or sit an agent out, then count the band in.")
        self.status.setWordWrap(True)
        bar = QHBoxLayout()
        bar.addWidget(self.count_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(self.status, 1)

        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(scroll, 1)
        self.setMinimumWidth(480)
        self.setMinimumHeight(260)

        self._left = 0
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._count)
        self._glow = QTimer(self)
        self._glow.setSingleShot(True)
        self._glow.timeout.connect(lambda: self._light(False))
        self.band_ready.connect(self._play)

    # -- the seats ---------------------------------------------------------
    def playing_seats(self) -> list[B.Seat]:
        return [s for s in self.seats if s.playing]

    def _seats_changed(self) -> None:
        names = ", ".join(f"{s.agent} ({'mic' if s.role == B.VOCAL else s.instrument.replace('_', ' ')})"
                          for s in self.playing_seats()) or "nobody"
        self.status.setText(f"On stage: {names}.")
        self.changed.emit()

    def _light(self, lit: bool) -> None:
        for card in self.cards:
            card.set_lit(lit)

    # -- count in, render, play -------------------------------------------
    def count_in(self) -> None:
        if not self.playing_seats():
            self.status.setText("Nobody is playing: tick at least one seat.")
            return
        self.count_btn.setEnabled(False)
        self._rendered = None
        seats = [B.Seat(**s.to_dict()) for s in self.seats]          # a snapshot: the worker never sees a seat change mid-render
        sr = self.app.sample_rate
        self.app.jobs.submit("band.render", "band", lambda ctx: B.render(seats, sr=sr),
                             on_done=self._rendered_ok, on_error=self._render_failed, description="Render the band")
        self._left = COUNTDOWN_S
        self._count()
        self._tick.start(1000)

    def _rendered_ok(self, audio) -> None:
        self._rendered = audio
        if self._left <= 0:
            self.band_ready.emit(audio)

    def _render_failed(self, exc) -> None:
        self._tick.stop()
        self.count_btn.setEnabled(True)
        self.status.setText(f"The band could not be rendered: {exc}")

    def _count(self) -> None:
        if self._left > 0:
            self.status.setText(f"Get ready... {self._left}")
            self._left -= 1
            return
        self._tick.stop()
        if self._rendered is not None:
            self.band_ready.emit(self._rendered)
        else:
            self.status.setText("Tuning up... (still rendering)")

    def _play(self, audio) -> None:
        self.app._cache_render("band", audio)
        self.app.play_render("band")
        self.count_btn.setEnabled(True)
        self._light(True)
        self._glow.start(int(1000 * len(audio) / self.app.sample_rate))
        self.status.setText("Playing: " + ", ".join(s.agent for s in self.playing_seats()))

    def stop(self) -> None:
        self._tick.stop()
        self._left = 0
        self._rendered = None
        self.count_btn.setEnabled(True)
        self.app.stop()
        self._light(False)
        self.status.setText("Stopped.")
