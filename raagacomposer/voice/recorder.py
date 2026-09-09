"""Direct take recording: Record, Stop, Cancel, and nothing in between.

The microphone is opened when the creator presses Record and released
when they press Stop or Cancel - there is no ambient capture, and the
speech-capture manager, which holds the input open for the whole session,
is a different thing (``speech.capture``).  What comes back from ``stop``
is the audio and nothing else: the controller decides where it belongs.

The input stream is injected, so the lifecycle and the shape of the audio
can be tested without a device: ``open_stream`` takes the same arguments
``sounddevice.InputStream`` does and returns an object with ``start``,
``stop`` and ``close``.  The default opens sounddevice, the backend the
application already installs, and says plainly when it is missing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from ..core.logging_setup import get_logger

log = get_logger("recorder")

try:  # pragma: no cover - machine dependent
    import sounddevice as _sd
except Exception as exc:  # noqa: BLE001
    _sd = None
    log.warning("sounddevice unavailable for recording: %s", exc)

#: What a Record button should say when there is nothing to record with.
NO_BACKEND = ("No audio input backend is available: the sounddevice package is "
              "not installed or could not load. Install it (pip install "
              "sounddevice) and restart, or check that an input device is "
              "present in the system's sound settings.")


def _sounddevice_stream(samplerate: int, channels: int, blocksize: int,
                        device, callback):
    if _sd is None:
        raise RuntimeError(NO_BACKEND)
    return _sd.InputStream(samplerate=samplerate, channels=channels,
                           dtype="float32", blocksize=blocksize,
                           device=device, callback=callback)


@dataclass
class RecorderState:
    #: ``off`` (no backend), ``ready``, ``recording``, ``error``.
    phase: str = "ready"
    error: str = ""
    device: str = ""
    started_at: float = 0.0
    seconds: float = 0.0
    level: float = 0.0


class TakeRecorder:
    """One take at a time, from an explicit start to an explicit stop."""

    def __init__(self, open_stream: Optional[Callable] = None,
                 sample_rate: int = 44100, block: int = 1024,
                 backend_present: Optional[bool] = None) -> None:
        self._open_stream = open_stream or _sounddevice_stream
        self.sample_rate = int(sample_rate)
        self.block = int(block)
        self._present = (backend_present if backend_present is not None
                         else (open_stream is not None or _sd is not None))
        self.state = RecorderState(phase="ready" if self._present else "off",
                                   error="" if self._present else NO_BACKEND)
        self._stream = None
        self._blocks: List[np.ndarray] = []
        self._lock = threading.Lock()
        #: Counts starts, so a result can be tied to the start that made it.
        self.session = 0

    # -- what is known ---------------------------------------------------
    @property
    def available(self) -> bool:
        return self._present

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        if not self.recording:
            return self.state.seconds
        return time.monotonic() - self.state.started_at

    # -- lifecycle -------------------------------------------------------
    def start(self, device=None) -> bool:
        """Open the input and begin keeping what it gives.  False, with the
        reason in ``state.error``, when it cannot."""
        if self.recording:
            return True
        if not self._present:
            self.state.phase = "off"
            self.state.error = NO_BACKEND
            return False
        with self._lock:
            self._blocks = []
        # Each capture attempt is its own session, numbered before the
        # stream is opened, and its callback carries that number: a block
        # from a stream that was stopped, cancelled or abandoned - one a
        # driver delivers late - is refused by number, not by whether some
        # stream happens to be open now.  A failed attempt keeps its
        # number too, so a callback it retained can never match the
        # successful retry that follows it.
        self.session += 1
        session = self.session

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            self._on_block(session, indata, status)
        stream = None
        try:
            stream = self._open_stream(samplerate=self.sample_rate, channels=1,
                                       blocksize=self.block, device=device,
                                       callback=callback)
            stream.start()
        except Exception as exc:  # noqa: BLE001 - a device problem is a state, not a crash
            self.state.phase = "error"
            self.state.error = (f"The input device ({device or 'default'}) could not "
                                f"be opened: {exc}. Check that it is connected and "
                                f"not in use by another application, or choose "
                                f"another input in Settings.")
            log.error(self.state.error)
            if stream is not None:
                # Constructed but never started: release it, whatever its
                # stop and close make of that, and keep the error above.
                for step in ("stop", "close"):
                    try:
                        getattr(stream, step)()
                    except Exception as cleanup:  # noqa: BLE001
                        log.warning("%s of an input that failed to start: %s", step, cleanup)
            return False
        self._stream = stream
        self.state.phase = "recording"
        self.state.error = ""
        self.state.device = str(device or "default")
        self.state.started_at = time.monotonic()
        self.state.seconds = 0.0
        self.state.level = 0.0
        log.info("recording take %d on %s", self.session, self.state.device)
        return True

    def stop(self) -> Optional[np.ndarray]:
        """Release the input and return what was recorded, mono float32 at
        ``sample_rate``.  None when nothing was being recorded."""
        if not self.recording:
            return None
        self._release()
        with self._lock:
            blocks, self._blocks = self._blocks, []
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        self.state.seconds = len(audio) / float(self.sample_rate)
        log.info("take %d stopped: %.2fs", self.session, self.state.seconds)
        return audio

    def cancel(self) -> None:
        """Release the input and keep nothing."""
        if not self.recording:
            return
        self._release()
        with self._lock:
            self._blocks = []
        self.state.seconds = 0.0
        log.info("take %d cancelled", self.session)

    def close(self) -> None:
        self.cancel()

    def _release(self) -> None:
        stream, self._stream = self._stream, None
        self.state.phase = "ready"
        if stream is None:
            return
        for step in ("stop", "close"):
            try:
                getattr(stream, step)()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s of the input stream failed: %s", step, exc)

    # -- audio thread ----------------------------------------------------
    def _on_block(self, session: int, indata, status) -> None:  # noqa: ANN001
        if status:
            log.debug("recording status: %s", status)
        chunk = np.array(np.asarray(indata)[:, 0], dtype=np.float32)
        with self._lock:
            if self._stream is None or session != self.session:
                return   # after Stop or Cancel, or from an earlier take's stream
            self._blocks.append(chunk)
        if len(chunk):
            self.state.level = float(np.sqrt(np.mean(chunk ** 2)))

    # -- for the screen --------------------------------------------------
    def status_text(self) -> str:
        if self.recording:
            return f"Recording, {self.seconds:.0f}s"
        if self.state.phase == "off":
            return "Input unavailable: " + self.state.error
        if self.state.phase == "error":
            return "Input error: " + self.state.error
        return "Not recording"
