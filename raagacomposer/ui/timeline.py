"""Timeline widget: ruler, named sections, tracks, regions, waveform, playhead.

Spec section 14G.  The creator can click to move the playhead, drag to select a
time range, click a region to select it, and right-click a region for the
region-scoped actions (regenerate, lock, remove, replace).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QPainter, QPen,
                           QPolygon)
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from ..core.models import ArrangementVersion, MelodyVersion, Region, Track
from ..music.theory import format_time_short
from . import theme

RULER_H = 26
SECTION_H = 22
TRACK_H = 44
HEADER_W = 150
MIN_PPS = 1.0
MAX_PPS = 60.0


class TimelineWidget(QWidget):
    seekRequested = Signal(float)
    selectionChanged = Signal(object)          # (start, end) or None
    regionSelected = Signal(str, str)          # track_id, region_id
    regionAction = Signal(str, str, str)       # action, track_id, region_id
    trackAction = Signal(str, str)             # action, track_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

        self.pps = 6.0
        self.duration = 60.0
        self.playhead = 0.0
        self.selection: Optional[Tuple[float, float]] = None
        self.melody: Optional[MelodyVersion] = None
        self.arrangement: Optional[ArrangementVersion] = None
        self.waveform: Optional[np.ndarray] = None
        self.waveform_duration = 0.0
        self.selected_region: Tuple[str, str] = ("", "")

        self._drag_start: Optional[float] = None
        self._dragging = False

    # -- data --------------------------------------------------------------
    def set_project(self, melody: Optional[MelodyVersion],
                    arrangement: Optional[ArrangementVersion],
                    duration: float) -> None:
        self.melody = melody
        self.arrangement = arrangement
        self.duration = max(10.0, float(duration or 0.0))
        self._resize_to_content()
        self.update()

    def set_waveform(self, audio: Optional[np.ndarray], sample_rate: int) -> None:
        if audio is None or len(audio) == 0:
            self.waveform = None
            self.waveform_duration = 0.0
        else:
            mono = audio if audio.ndim == 1 else audio.mean(axis=1)
            buckets = 4000
            step = max(1, len(mono) // buckets)
            trimmed = mono[:step * (len(mono) // step)]
            if len(trimmed) == 0:
                self.waveform = None
                return
            reshaped = trimmed.reshape(-1, step)
            self.waveform = np.abs(reshaped).max(axis=1)
            self.waveform_duration = len(mono) / float(sample_rate)
        self.update()

    def set_playhead(self, seconds: float) -> None:
        if abs(seconds - self.playhead) < 0.02:
            return
        self.playhead = max(0.0, seconds)
        self.update()

    def set_selection(self, selection: Optional[Tuple[float, float]]) -> None:
        self.selection = selection
        self.update()

    def zoom(self, factor: float) -> None:
        self.pps = max(MIN_PPS, min(MAX_PPS, self.pps * factor))
        self._resize_to_content()
        self.update()

    def zoom_to_fit(self, viewport_width: int) -> None:
        usable = max(120, viewport_width - HEADER_W - 20)
        self.pps = max(MIN_PPS, min(MAX_PPS, usable / max(1.0, self.duration)))
        self._resize_to_content()
        self.update()

    # -- geometry ----------------------------------------------------------
    def _resize_to_content(self) -> None:
        width = int(HEADER_W + self.duration * self.pps + 40)
        tracks = len(self.arrangement.tracks) if self.arrangement else 0
        height = RULER_H + SECTION_H + max(1, tracks) * TRACK_H + 30
        self.setMinimumSize(QSize(width, max(220, height)))
        self.resize(width, max(self.height(), height))

    def x_for(self, seconds: float) -> int:
        return int(HEADER_W + seconds * self.pps)

    def time_at(self, x: int) -> float:
        return max(0.0, (x - HEADER_W) / max(0.001, self.pps))

    def _track_rect(self, index: int) -> QRect:
        top = RULER_H + SECTION_H + index * TRACK_H
        return QRect(0, top, self.width(), TRACK_H)

    def _hit_region(self, pos: QPoint) -> Tuple[Optional[Track], Optional[Region]]:
        if not self.arrangement or pos.x() < HEADER_W:
            return None, None
        for i, track in enumerate(self.arrangement.tracks):
            rect = self._track_rect(i)
            if not rect.contains(QPoint(pos.x(), pos.y())):
                continue
            t = self.time_at(pos.x())
            for region in track.regions:
                if region.start <= t <= region.end:
                    return track, region
            return track, None
        return None, None

    def _hit_track(self, pos: QPoint) -> Optional[Track]:
        if not self.arrangement:
            return None
        for i, track in enumerate(self.arrangement.tracks):
            if self._track_rect(i).contains(QPoint(pos.x(), pos.y())):
                return track
        return None

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), theme.color("bg"))

        self._paint_sections(p)
        self._paint_selection(p)
        self._paint_waveform(p)
        self._paint_tracks(p)
        self._paint_ruler(p)
        self._paint_playhead(p)
        p.end()

    def _paint_ruler(self, p: QPainter) -> None:
        rect = QRect(0, 0, self.width(), RULER_H)
        p.fillRect(rect, theme.color("panel"))
        p.setPen(QPen(theme.color("border")))
        p.drawLine(0, RULER_H - 1, self.width(), RULER_H - 1)

        step = 60.0
        for candidate in (1, 2, 5, 10, 15, 30, 60, 120):
            if candidate * self.pps >= 55:
                step = float(candidate)
                break
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        t = 0.0
        while t <= self.duration + step:
            x = self.x_for(t)
            if x > self.width():
                break
            p.setPen(QPen(theme.color("muted")))
            p.drawLine(x, RULER_H - 8, x, RULER_H - 1)
            p.drawText(x + 3, RULER_H - 10, format_time_short(t))
            t += step
        p.setPen(QPen(theme.color("text")))
        p.drawText(6, RULER_H - 8, "Timeline")

    def _paint_sections(self, p: QPainter) -> None:
        top = RULER_H
        p.fillRect(QRect(0, top, self.width(), SECTION_H), theme.color("panel_alt"))
        if not self.melody:
            return
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        metrics = QFontMetrics(font)
        for section in self.melody.sections:
            x0 = self.x_for(section.start)
            x1 = self.x_for(section.end)
            rect = QRect(x0, top, max(2, x1 - x0), SECTION_H)
            p.fillRect(rect, theme.section_color(section.kind.value))
            p.setPen(QPen(theme.color("border")))
            p.drawRect(rect)
            label = section.name + (" [locked]" if section.locked else "")
            if metrics.horizontalAdvance(label) < rect.width() - 6:
                p.setPen(QPen(theme.color("text")))
                p.drawText(rect.adjusted(4, 0, -2, 0),
                           Qt.AlignVCenter | Qt.AlignLeft, label)
        # Full-height section dividers.
        p.setPen(QPen(QColor(255, 255, 255, 16)))
        for section in self.melody.sections:
            x = self.x_for(section.start)
            p.drawLine(x, top, x, self.height())

    def _paint_waveform(self, p: QPainter) -> None:
        if self.waveform is None or self.waveform_duration <= 0:
            return
        top = RULER_H + SECTION_H
        height = self.height() - top - 4
        if height <= 8:
            return
        mid = top + height / 2
        peak = float(self.waveform.max()) or 1.0
        p.setPen(QPen(QColor(224, 164, 88, 40)))
        n = len(self.waveform)
        for x in range(HEADER_W, self.width()):
            t = self.time_at(x)
            if t > self.waveform_duration:
                break
            idx = min(n - 1, int(t / self.waveform_duration * n))
            amp = float(self.waveform[idx]) / peak * (height / 2 - 4)
            p.drawLine(x, int(mid - amp), x, int(mid + amp))

    def _paint_selection(self, p: QPainter) -> None:
        if not self.selection:
            return
        start, end = self.selection
        x0, x1 = self.x_for(min(start, end)), self.x_for(max(start, end))
        rect = QRect(x0, RULER_H, max(1, x1 - x0), self.height() - RULER_H)
        p.fillRect(rect, QColor(58, 74, 99, 90))
        p.setPen(QPen(theme.color("selection")))
        p.drawRect(rect)

    def _paint_tracks(self, p: QPainter) -> None:
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        tracks = self.arrangement.tracks if self.arrangement else []
        if not tracks:
            p.setPen(QPen(theme.color("muted")))
            p.drawText(QRect(HEADER_W + 12, RULER_H + SECTION_H + 16, 460, 24),
                       Qt.AlignLeft,
                       "No instruments yet - say \"add veena here\" or use the "
                       "Arrangement controls.")
            return

        for i, track in enumerate(tracks):
            rect = self._track_rect(i)
            p.fillRect(rect, theme.color("panel") if i % 2 == 0
                       else theme.color("panel_alt"))
            p.setPen(QPen(theme.color("border")))
            p.drawLine(0, rect.bottom(), self.width(), rect.bottom())

            # header
            header = QRect(0, rect.top(), HEADER_W, rect.height())
            p.fillRect(header, theme.color("panel_alt"))
            p.setPen(QPen(theme.color("border")))
            p.drawLine(HEADER_W, rect.top(), HEADER_W, rect.bottom())
            p.setPen(QPen(theme.color("text")))
            p.drawText(header.adjusted(8, 4, -4, -20), Qt.AlignLeft | Qt.AlignVCenter,
                       track.label)
            flags = []
            if track.mute:
                flags.append("M")
            if track.solo:
                flags.append("S")
            if track.locked:
                flags.append("LOCK")
            p.setPen(QPen(theme.color("muted")))
            p.drawText(header.adjusted(8, 20, -4, -2), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{track.role}  {' '.join(flags)}")

            for region in track.regions:
                x0 = self.x_for(region.start)
                x1 = self.x_for(region.end)
                r = QRect(x0, rect.top() + 6, max(3, x1 - x0), rect.height() - 14)
                base = theme.role_color(region.role)
                if track.mute:
                    base = base.darker(180)
                p.fillRect(r, QBrush(base.darker(140)))
                selected = (track.id, region.id) == self.selected_region
                p.setPen(QPen(theme.color("accent") if selected else base.lighter(130),
                              2 if selected else 1))
                p.drawRect(r)
                if region.locked:
                    p.fillRect(QRect(r.left(), r.top(), 4, r.height()),
                               theme.color("accent"))
                if r.width() > 60:
                    p.setPen(QPen(theme.color("text")))
                    p.drawText(r.adjusted(8, 0, -4, 0),
                               Qt.AlignVCenter | Qt.AlignLeft,
                               f"{len(region.notes)} notes")
                # note blocks
                if r.width() > 24 and region.notes:
                    lows = [n.midi for n in region.notes]
                    lo, hi = min(lows), max(lows) or 1
                    span = max(1, hi - lo)
                    p.setPen(QPen(QColor(255, 255, 255, 70)))
                    for note in region.notes[:900]:
                        nx0 = self.x_for(note.start)
                        nx1 = max(nx0 + 1, self.x_for(note.end))
                        ny = r.bottom() - 3 - int((note.midi - lo) / span *
                                                  (r.height() - 8))
                        p.drawLine(nx0, ny, nx1, ny)

    def _paint_playhead(self, p: QPainter) -> None:
        x = self.x_for(self.playhead)
        p.setPen(QPen(theme.color("playhead"), 1))
        p.drawLine(x, 0, x, self.height())
        head = QPolygon([QPoint(x - 5, 0), QPoint(x + 5, 0), QPoint(x, 9)])
        p.setBrush(QBrush(theme.color("playhead")))
        p.setPen(Qt.NoPen)
        p.drawPolygon(head)

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        pos = event.position().toPoint()
        if event.button() != Qt.LeftButton:
            return
        if pos.x() < HEADER_W:
            track = self._hit_track(pos)
            if track:
                self.trackAction.emit("select", track.id)
            return
        t = self.time_at(pos.x())
        if pos.y() <= RULER_H + SECTION_H:
            self._drag_start = t
            self._dragging = True
            self.seekRequested.emit(t)
            return
        track, region = self._hit_region(pos)
        if track and region:
            self.selected_region = (track.id, region.id)
            self.regionSelected.emit(track.id, region.id)
            self.update()
        self._drag_start = t
        self._dragging = True

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if not self._dragging or self._drag_start is None:
            return
        t = self.time_at(event.position().toPoint().x())
        if abs(t - self._drag_start) > 0.15:
            self.selection = (min(self._drag_start, t), max(self._drag_start, t))
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._dragging:
            self._dragging = False
            if self.selection:
                self.selectionChanged.emit(self.selection)
            else:
                self.selectionChanged.emit(None)
        self._drag_start = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        self.selection = None
        self.selectionChanged.emit(None)
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        if event.modifiers() & Qt.ControlModifier:
            self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
            event.accept()
        else:
            event.ignore()

    def contextMenuEvent(self, event) -> None:  # noqa: ANN001
        pos = event.pos()
        track, region = self._hit_region(pos)
        menu = QMenu(self)
        if track and region:
            self.selected_region = (track.id, region.id)
            self.update()
            menu.addAction("Regenerate this region", lambda: self.regionAction.emit(
                "regenerate", track.id, region.id))
            menu.addAction("Unlock region" if region.locked else "Lock region",
                           lambda: self.regionAction.emit(
                               "unlock" if region.locked else "lock",
                               track.id, region.id))
            menu.addAction("Replace instrument here...",
                           lambda: self.regionAction.emit("replace", track.id, region.id))
            menu.addAction("Remove this region",
                           lambda: self.regionAction.emit("remove", track.id, region.id))
            menu.addSeparator()
        if track:
            menu.addAction("Unmute track" if track.mute else "Mute track",
                           lambda: self.trackAction.emit("mute", track.id))
            menu.addAction("Unsolo track" if track.solo else "Solo track",
                           lambda: self.trackAction.emit("solo", track.id))
            menu.addAction("Unlock track" if track.locked else "Lock track",
                           lambda: self.trackAction.emit("lock", track.id))
            menu.addAction("Remove instrument",
                           lambda: self.trackAction.emit("remove", track.id))
            menu.addSeparator()
        menu.addAction("Add instrument here...",
                       lambda: self.trackAction.emit("add", ""))
        menu.addAction("Clear selection", lambda: (
            self.set_selection(None), self.selectionChanged.emit(None)))
        menu.exec(event.globalPos())
