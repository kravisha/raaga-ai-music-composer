"""Speech-to-text adapters (spec sections 5.1, 11).

The rest of the application only sees :class:`STTAdapter`.  Three concrete
adapters ship:

``VoskSTT``      offline streaming recognition when the ``vosk`` package and a
                 model directory are present;
``WhisperSTT``   local batch recognition of each captured utterance when
                 ``faster-whisper`` or ``openai-whisper`` is installed;
``TypedSTT``     the always-available fallback -- text typed into the
                 conversation panel enters the identical command pipeline, so
                 no part of the workflow depends on a cloud key.

A cloud streaming provider is added by writing one more subclass.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..core.logging_setup import get_logger
from ..core.settings import Settings, config_dir

log = get_logger("stt")


@dataclass
class Transcript:
    text: str
    final: bool = True
    confidence: float = 1.0
    source: str = ""


class STTAdapter:
    """Interface every speech backend implements."""

    name = "none"
    streaming = False

    @property
    def available(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def feed(self, audio: np.ndarray, sample_rate: int) -> Optional[Transcript]:
        """Push audio; may return a partial transcript."""
        return None

    def finish(self) -> Optional[Transcript]:
        """End of utterance; return the final transcript."""
        return None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Optional[Transcript]:
        """Batch transcription of one complete utterance."""
        self.reset()
        self.feed(audio, sample_rate)
        return self.finish()

    def status(self) -> str:
        return f"{self.name}: {'ready' if self.available else 'unavailable'}"


class TypedSTT(STTAdapter):
    """Typed input treated as a final transcript."""

    name = "typed"

    @property
    def available(self) -> bool:
        return True

    def status(self) -> str:
        return "typed input (no speech backend configured)"


class VoskSTT(STTAdapter):
    name = "vosk"
    streaming = True

    def __init__(self, model_dir: Optional[Path] = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._rec = None
        self._model = None
        self._error = ""
        self.model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self._load()

    @staticmethod
    def _default_model_dir() -> Optional[Path]:
        base = config_dir() / "models"
        if base.exists():
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / "am").exists():
                    return child
                if child.is_dir() and any(child.glob("**/final.mdl")):
                    return child
        return None

    def _load(self) -> None:
        if not self.model_dir or not Path(self.model_dir).exists():
            self._error = ("No Vosk model found. Place one under "
                           f"{config_dir() / 'models'}.")
            return
        try:
            import vosk  # type: ignore
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(self.model_dir))
            self._rec = vosk.KaldiRecognizer(self._model, self.sample_rate)
            log.info("vosk model loaded from %s", self.model_dir)
        except Exception as exc:  # noqa: BLE001
            self._error = f"Vosk unavailable: {exc}"
            log.warning(self._error)

    @property
    def available(self) -> bool:
        return self._rec is not None

    def reset(self) -> None:
        if self._model is not None:
            import vosk  # type: ignore
            self._rec = vosk.KaldiRecognizer(self._model, self.sample_rate)

    def feed(self, audio: np.ndarray, sample_rate: int) -> Optional[Transcript]:
        if self._rec is None:
            return None
        pcm = _to_pcm16(audio, sample_rate, self.sample_rate)
        try:
            if self._rec.AcceptWaveform(pcm):
                text = json.loads(self._rec.Result()).get("text", "")
                return Transcript(text, True, source=self.name) if text else None
            partial = json.loads(self._rec.PartialResult()).get("partial", "")
            return Transcript(partial, False, source=self.name) if partial else None
        except Exception as exc:  # noqa: BLE001
            log.warning("vosk feed failed: %s", exc)
            return None

    def finish(self) -> Optional[Transcript]:
        if self._rec is None:
            return None
        try:
            text = json.loads(self._rec.FinalResult()).get("text", "")
        except Exception:  # noqa: BLE001
            text = ""
        self.reset()
        return Transcript(text, True, source=self.name) if text else None

    def status(self) -> str:
        return f"vosk: ready ({self.model_dir})" if self.available else \
            f"vosk: {self._error}"


class WhisperSTT(STTAdapter):
    name = "whisper"
    streaming = False

    def __init__(self, model_size: str = "base") -> None:
        """Find a Whisper, but do not load it yet.

        ``build_adapter`` constructs every candidate backend to ask which
        one works, and it runs while the application is starting.  Loading
        the model here cost fifteen seconds of startup for everyone,
        whether or not they ever pressed the microphone - and a creator who
        never uses voice paid it every time.

        So this only establishes that a Whisper is *installed*.  The model
        itself is loaded on the first phrase, on the capture thread, where
        a pause is expected because you have just finished speaking.
        """
        self._model = None
        self._model_size = model_size
        self._kind = ""
        self._error = ""
        self._buffer: List[np.ndarray] = []
        self._sr = 16000
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            self._kind = "faster-whisper"
        except Exception:
            try:
                import whisper  # type: ignore  # noqa: F401
                self._kind = "openai-whisper"
            except Exception as exc:  # noqa: BLE001
                self._error = f"Whisper unavailable: {exc}"
        if self._kind:
            log.info("whisper available (%s, %s); the model loads on the "
                     "first phrase", self._kind, model_size)

    def _ensure_model(self):
        """Load on first use.  A failure here is reported, never raised."""
        if self._model is not None or not self._kind:
            return self._model
        try:
            if self._kind == "faster-whisper":
                from faster_whisper import WhisperModel  # type: ignore
                self._model = WhisperModel(self._model_size, device="cpu",
                                           compute_type="int8")
            else:
                import whisper  # type: ignore
                self._model = whisper.load_model(self._model_size)
            log.info("whisper model loaded (%s, %s)", self._kind,
                     self._model_size)
        except Exception as exc:  # noqa: BLE001
            self._error = f"Whisper model could not be loaded: {exc}"
            log.error(self._error)
            self._kind = ""
        return self._model

    @property
    def available(self) -> bool:
        """Whether a Whisper is installed - not whether it is loaded."""
        return bool(self._kind)

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, audio: np.ndarray, sample_rate: int) -> Optional[Transcript]:
        self._sr = sample_rate
        self._buffer.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return None

    def finish(self) -> Optional[Transcript]:
        if not self._buffer or self._ensure_model() is None:
            self._buffer.clear()
            return None
        audio = np.concatenate(self._buffer)
        self._buffer.clear()
        audio = _resample(audio, self._sr, 16000)
        try:
            if self._kind == "faster-whisper":
                segments, _ = self._model.transcribe(audio, language=None)
                text = " ".join(s.text for s in segments).strip()
            else:
                text = str(self._model.transcribe(audio).get("text", "")).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("whisper transcription failed: %s", exc)
            return None
        return Transcript(text, True, source=self.name) if text else None

    def status(self) -> str:
        if not self.available:
            return f"whisper: {self._error}"
        loaded = "ready" if self._model is not None else "ready, loads on first use"
        return f"whisper: {loaded} ({self._kind})"


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src == dst or len(audio) == 0:
        return audio
    n = int(round(len(audio) * dst / src))
    return np.interp(np.linspace(0, len(audio) - 1, n),
                     np.arange(len(audio)), audio).astype(np.float32)


def _to_pcm16(audio: np.ndarray, src: int, dst: int) -> bytes:
    resampled = _resample(audio, src, dst)
    clipped = np.clip(resampled, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def build_adapter(settings: Optional[Settings] = None) -> STTAdapter:
    """Pick a speech backend from settings, falling back to typed input."""
    settings = settings or Settings.load()
    choice = (settings.stt_provider or "auto").lower()
    if choice in ("none", "typed"):
        return TypedSTT()
    candidates: List[STTAdapter] = []
    if choice in ("auto", "vosk"):
        candidates.append(VoskSTT())
    if choice in ("auto", "whisper"):
        candidates.append(WhisperSTT(
            str(getattr(settings, "stt_model_size", "tiny") or "tiny")))
    for adapter in candidates:
        if adapter.available:
            log.info("speech backend: %s", adapter.status())
            return adapter
    log.info("no speech backend available; typed input is active")
    return TypedSTT()
