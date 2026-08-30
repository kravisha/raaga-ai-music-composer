"""Playback engine (spec sections 6, 12.9, 18).

Wraps a sounddevice output stream with the operations the natural-language
timeline commands need: play a range, pause, resume, stop, seek, loop, and
report the playhead continuously.  Audio-device failures are reported rather
than raised into the UI, and the engine can be re-opened after a device is
unplugged and plugged back in.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import numpy as np

from ..core.logging_setup import get_logger

log = get_logger("playback")

try:  # pragma: no cover - depends on the machine's audio stack
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001
    sd = None
    log.warning("sounddevice unavailable: %s", exc)


class PlaybackEngine:
    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self._buffer = np.zeros((0, 2), dtype=np.float32)
        self._pos = 0
        self._start = 0
        self._end = 0
        self._loop = False
        self._playing = False
        self._paused = False
        self._stream = None
        self._volume = 1.0
        self.last_error = ""
        self.source_name = ""
        self.on_finished: Optional[Callable[[], None]] = None

    # -- state -------------------------------------------------------------
    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def playing(self) -> bool:
        return self._playing and not self._paused

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def duration(self) -> float:
        return len(self._buffer) / self.sample_rate if len(self._buffer) else 0.0

    @property
    def position(self) -> float:
        with self._lock:
            return self._pos / self.sample_rate

    @property
    def range(self) -> Tuple[float, float]:
        return (self._start / self.sample_rate, self._end / self.sample_rate)

    def set_volume(self, value: float) -> None:
        self._volume = max(0.0, min(2.0, float(value)))

    # -- content -----------------------------------------------------------
    def load(self, audio: np.ndarray, sample_rate: Optional[int] = None,
             name: str = "") -> None:
        with self._lock:
            was_playing = self.playing
        self.stop()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        with self._lock:
            self._buffer = np.ascontiguousarray(audio[:, :2], dtype=np.float32)
            self.sample_rate = int(sample_rate or self.sample_rate)
            self._pos = 0
            self._start = 0
            self._end = len(self._buffer)
            self.source_name = name
        log.info("loaded %.1fs of audio (%s)", self.duration, name or "unnamed")
        if was_playing:
            self.play()

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._buffer = np.zeros((0, 2), dtype=np.float32)
            self.source_name = ""

    # -- transport ---------------------------------------------------------
    def play(self, start: Optional[float] = None, end: Optional[float] = None,
             loop: bool = False) -> bool:
        if not self.available:
            self.last_error = "No audio output backend is available."
            return False
        with self._lock:
            if len(self._buffer) == 0:
                self.last_error = "Nothing has been rendered to play yet."
                return False
            sr = self.sample_rate
            self._start = 0 if start is None else max(0, int(start * sr))
            self._end = len(self._buffer) if end is None else \
                min(len(self._buffer), int(end * sr))
            if self._end <= self._start:
                self._end = len(self._buffer)
            self._pos = self._start
            self._loop = loop
            self._paused = False
        return self._open_stream()

    def play_from(self, start: float, loop: bool = False) -> bool:
        return self.play(start, None, loop)

    def pause(self) -> None:
        with self._lock:
            if self._playing:
                self._paused = True
        self._close_stream(keep_position=True)

    def resume(self) -> bool:
        with self._lock:
            if not self._paused or len(self._buffer) == 0:
                return False
            self._paused = False
        return self._open_stream()

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
            return False
        if self._paused:
            return self.resume()
        return self.play()

    def stop(self) -> None:
        self._close_stream(keep_position=False)
        with self._lock:
            self._playing = False
            self._paused = False
            self._pos = self._start

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._pos = max(0, min(len(self._buffer),
                                   int(seconds * self.sample_rate)))
            if self._pos >= self._end:
                self._end = len(self._buffer)

    def nudge(self, delta_seconds: float) -> None:
        self.seek(self.position + delta_seconds)

    # -- stream plumbing ---------------------------------------------------
    def _open_stream(self) -> bool:
        self._close_stream(keep_position=True)
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=2, dtype="float32",
                blocksize=1024, callback=self._callback,
                finished_callback=self._on_stream_finished)
            self._stream.start()
            self._playing = True
            self.last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Audio device error: {exc}"
            log.error(self.last_error)
            self._stream = None
            self._playing = False
            return False

    def _close_stream(self, keep_position: bool) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("closing stream failed: %s", exc)
        if not keep_position:
            with self._lock:
                self._pos = self._start

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.debug("playback status: %s", status)
        with self._lock:
            if self._paused or len(self._buffer) == 0:
                outdata[:] = 0
                return
            remaining = self._end - self._pos
            if remaining <= 0:
                if self._loop:
                    self._pos = self._start
                    remaining = self._end - self._pos
                else:
                    outdata[:] = 0
                    raise sd.CallbackStop
            take = min(frames, remaining)
            chunk = self._buffer[self._pos:self._pos + take] * self._volume
            outdata[:take] = chunk
            if take < frames:
                if self._loop:
                    self._pos = self._start
                    rest = min(frames - take, self._end - self._pos)
                    outdata[take:take + rest] = \
                        self._buffer[self._pos:self._pos + rest] * self._volume
                    self._pos += rest
                    if take + rest < frames:
                        outdata[take + rest:] = 0
                    return
                outdata[take:] = 0
            self._pos += take

    def _on_stream_finished(self) -> None:
        with self._lock:
            still_paused = self._paused
        if still_paused:
            return
        self._playing = False
        cb = self.on_finished
        if cb:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.debug("on_finished callback failed", exc_info=True)

    def close(self) -> None:
        self._close_stream(keep_position=False)


def output_devices() -> list:
    if sd is None:
        return []
    try:
        return [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot query output devices: %s", exc)
        return []


def input_devices() -> list:
    if sd is None:
        return []
    try:
        return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot query input devices: %s", exc)
        return []
