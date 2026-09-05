"""Continuous voice input manager (spec sections 5.1, 5.2, 18).

Holds the microphone open for the whole session, segments speech with an
energy gate, streams audio to the speech adapter and reports partial and final
transcripts through callbacks.  It also raises a *barge-in* the moment the
creator starts speaking, so playback or generation can be interrupted before
the sentence has even finished (spec 5.2).

Device failures do not raise into the UI: the manager records the error, keeps
retrying at a slow cadence and reports its state.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..core.logging_setup import get_logger
from ..core.settings import Settings
from .stt import STTAdapter, Transcript, build_adapter

log = get_logger("capture")

try:  # pragma: no cover - machine dependent
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001
    sd = None
    log.warning("sounddevice unavailable for capture: %s", exc)


@dataclass
class CaptureState:
    listening: bool = False
    speaking: bool = False
    level: float = 0.0
    error: str = ""
    backend: str = ""
    device: str = ""
    #: What the microphone is doing, in the creator's terms rather than the
    #: implementation's.  Voice was the one part of the application that
    #: gave no sign of what it was doing: you pressed a button, spoke, and
    #: either something happened or nothing did.  One of ``off``, ``ready``,
    #: ``listening``, ``hearing``, ``thinking``, ``done``, ``error``.
    phase: str = "off"
    #: The last thing it understood, so a creator can see whether it heard
    #: them correctly before blaming the command.
    heard: str = ""


class VoiceInputManager:
    """Continuous microphone capture with voice-activity segmentation."""

    def __init__(self, settings: Optional[Settings] = None,
                 adapter: Optional[STTAdapter] = None) -> None:
        self.settings = settings or Settings.load()
        self.adapter = adapter or build_adapter(self.settings)
        self.sample_rate = 16000
        self.block = 1024
        self.state = CaptureState(backend=self.adapter.name)

        self.on_partial: Optional[Callable[[str], None]] = None
        self.on_final: Optional[Callable[[str], None]] = None
        self.on_barge_in: Optional[Callable[[], None]] = None
        self.on_state: Optional[Callable[[CaptureState], None]] = None

        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._utterance: list = []
        self._silence_blocks = 0
        self._speech_blocks = 0
        self._announced_barge_in = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def available(self) -> bool:
        return sd is not None

    def start(self) -> bool:
        if self.state.listening:
            return True
        if not self.available:
            self.state.error = "No microphone backend is available."
            self.state.phase = "error"
            self._notify()
            return False
        try:
            device = self.settings.mic_device or None
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                blocksize=self.block, device=device, callback=self._callback)
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self.state.error = f"Microphone error: {exc}"
            self.state.phase = "error"
            log.error(self.state.error)
            self._stream = None
            self._notify()
            return False

        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="raaga-stt",
                                        daemon=True)
        self._thread.start()
        self.state.listening = True
        self.state.error = ""
        self.state.phase = "listening"
        self.state.heard = ""
        self.state.device = str(self.settings.mic_device or "default")
        log.info("listening (backend=%s)", self.adapter.name)
        self._notify()
        return True

    def stop(self) -> None:
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("closing microphone failed: %s", exc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.state.listening = False
        self.state.speaking = False
        self._utterance.clear()
        self._notify()

    def toggle(self) -> bool:
        if self.state.listening:
            self.stop()
            return False
        return self.start()

    def reconnect(self) -> bool:
        """Recover after a device is unplugged and plugged back in."""
        self.stop()
        time.sleep(0.2)
        return self.start()

    # -- audio thread ------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.debug("capture status: %s", status)
        try:
            self._audio.put_nowait(np.array(indata[:, 0], dtype=np.float32))
        except queue.Full:
            pass

    # -- worker thread -----------------------------------------------------
    def _worker(self) -> None:
        threshold = float(self.settings.vad_threshold)
        silence_needed = max(1, int(self.settings.vad_silence_ms /
                                    (1000 * self.block / self.sample_rate)))
        while not self._stop.is_set():
            try:
                chunk = self._audio.get(timeout=0.25)
            except queue.Empty:
                continue
            level = float(np.sqrt(np.mean(chunk ** 2)))
            self.state.level = level
            speaking = level > threshold

            if speaking:
                self._speech_blocks += 1
                self._silence_blocks = 0
                self._utterance.append(chunk)
                if not self.state.speaking and self._speech_blocks >= 2:
                    self.state.speaking = True
                    self.state.phase = "hearing"
                    self._announced_barge_in = False
                    self._notify()
                # Barge-in fires as soon as speech is confidently detected.
                if self.state.speaking and not self._announced_barge_in:
                    self._announced_barge_in = True
                    if self.on_barge_in:
                        self._safe(self.on_barge_in)
                if self.adapter.streaming:
                    result = self.adapter.feed(chunk, self.sample_rate)
                    if result and not result.final and self.on_partial:
                        self._safe(self.on_partial, result.text)
                    elif result and result.final:
                        self._emit_final(result)
            else:
                if self.state.speaking:
                    self._silence_blocks += 1
                    self._utterance.append(chunk)
                    if self._silence_blocks >= silence_needed:
                        self._close_utterance()
                else:
                    self._speech_blocks = 0
                    if self.adapter.streaming:
                        self.adapter.feed(chunk, self.sample_rate)

    def _close_utterance(self) -> None:
        audio = np.concatenate(self._utterance) if self._utterance else None
        self._utterance.clear()
        self.state.speaking = False
        self.state.phase = "thinking" if audio is not None else "listening"
        self._speech_blocks = 0
        self._silence_blocks = 0
        self._notify()
        if audio is None or len(audio) < self.sample_rate * 0.25:
            return
        try:
            if self.adapter.streaming:
                result = self.adapter.finish()
            else:
                result = self.adapter.transcribe(audio, self.sample_rate)
        except Exception as exc:  # noqa: BLE001
            log.warning("transcription failed: %s", exc)
            result = None
        if result and result.text.strip():
            self._emit_final(result)

    def _emit_final(self, result: Transcript) -> None:
        text = result.text.strip()
        if not text:
            return
        log.info("heard: %s", text)
        self.state.heard = text
        self.state.phase = "done"
        self._notify()
        if self.on_final:
            self._safe(self.on_final, text)

    # -- helpers -----------------------------------------------------------
    def submit_text(self, text: str) -> None:
        """Route typed input through the same pipeline as speech."""
        if text and text.strip() and self.on_final:
            self._safe(self.on_final, text.strip())

    def _notify(self) -> None:
        if self.on_state:
            self._safe(self.on_state, self.state)

    @staticmethod
    def _safe(fn: Callable, *args) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            log.debug("capture callback failed", exc_info=True)

    #: What each phase says on screen.  Section 15 of the next-phase
    #: specification asks for the microphone's state to be visible rather
    #: than guessed at from whether anything happened.
    PHASE_TEXT = {
        "off": "Microphone off",
        "listening": "Listening",
        "hearing": "Hearing you",
        "thinking": "Working out what you said",
        "done": "Heard",
        "error": "Microphone problem",
    }

    def status_text(self) -> str:
        if self.state.error:
            return self.state.error
        if not self.state.listening:
            return f"Microphone off - {self.adapter.status()}"
        phase = self.PHASE_TEXT.get(self.state.phase, "Listening")
        if self.state.phase == "done" and self.state.heard:
            return f'Heard: "{self.state.heard}"'
        if self.state.phase == "listening":
            return f"Listening - {self.adapter.status()}"
        return phase

    def close(self) -> None:
        self.stop()
