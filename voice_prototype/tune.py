"""Melody controls to a continuous pitch curve — contract `5ee63d7` §1.3.

Two ways in, and the order matters. **Direct melody controls are preferred**:
where the melody is known symbolically, take it that way. Re-extracting f0 from
audio that something already knows symbolically is a lossy round trip we would be
choosing.

A monophonic WAV adapter exists for the case where only audio is available. It is
marked as an estimate wherever its output travels, because it is one.

Either way the result is a pitch curve that is never quantised to semitones, and
an explicit voicing mask in which **a rest is unvoiced and stays unvoiced**.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

HOP_SECONDS = 0.010
A4_HZ = 440.0
A4_MIDI = 69

#: Sources of a curve, recorded so a reader can tell measurement from statement.
DIRECT = "direct_melody_controls"
WAV_ADAPTER = "monophonic_wav_adapter"


class TuneError(Exception):
    """A refusal. Never a silent correction."""


def midi_to_hz(midi: float) -> float:
    return A4_HZ * 2.0 ** ((float(midi) - A4_MIDI) / 12.0)


@dataclass
class Event:
    """One melody event. A rest is an event too - that is the point.

    `midi` may be fractional: the curve is never snapped to a semitone, and a
    gamaka written as a sequence of fractional targets survives as written.
    """
    start_s: float
    duration_s: float
    midi: float | None = None          #: None means a rest
    #: Optional shape inside the event, as (offset_seconds, cents) pairs.
    #: This is how ornament detail is expressed in direct controls.
    bend: list = field(default_factory=list)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def is_rest(self) -> bool:
        return self.midi is None


@dataclass
class Tune:
    """A pitch curve, a voicing mask, and the events that produced them."""
    hop_s: float
    f0_hz: list                        #: one value per hop; 0.0 where unvoiced
    voiced: list                       #: one bool per hop
    events: list                       #: the events as dicts, rests included
    source: str                        #: DIRECT or WAV_ADAPTER
    is_estimate: bool                  #: true for the adapter, false for controls
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.f0_hz) * self.hop_s

    @property
    def sung_events(self) -> list:
        return [e for e in self.events if e.get("midi") is not None]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Tune":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _validate(events: list) -> list:
    if not events:
        raise TuneError("A tune with no events is not a tune.")
    ordered = sorted(events, key=lambda e: e.start_s)
    previous_end = None
    for index, event in enumerate(ordered):
        for name, value in (("start_s", event.start_s),
                            ("duration_s", event.duration_s)):
            if not isinstance(value, (int, float)) or value != value or math.isinf(value):
                raise TuneError(f"Event {index} has a non-finite {name}: {value!r}")
        if event.start_s < 0:
            raise TuneError(f"Event {index} starts before zero.")
        if event.duration_s <= 0:
            raise TuneError(f"Event {index} has no duration.")
        if event.midi is not None:
            if not isinstance(event.midi, (int, float)) or event.midi != event.midi:
                raise TuneError(f"Event {index} has a non-finite midi value.")
            if not 0 < float(event.midi) < 128:
                raise TuneError(f"Event {index} midi {event.midi!r} is out of range.")
        if previous_end is not None and event.start_s + 1e-9 < previous_end:
            raise TuneError(
                f"Event {index} starts at {event.start_s} while the previous one "
                f"runs to {previous_end}. Overlapping events have no single pitch.")
        previous_end = event.end_s
    return ordered


def from_events(events: list, *, hop_s: float = HOP_SECONDS,
                notes: str = "") -> Tune:
    """Direct melody controls to a curve. The preferred path.

    Gaps between events are rests, and so are explicit rest events. Both come
    out unvoiced, and nothing downstream is permitted to fill them.
    """
    ordered = _validate(events)
    total = max(e.end_s for e in ordered)
    count = int(math.ceil(total / hop_s))
    f0 = np.zeros(count, dtype=float)
    voiced = np.zeros(count, dtype=bool)

    for event in ordered:
        if event.is_rest:
            continue                      # silence is not a pitch to write
        begin = int(round(event.start_s / hop_s))
        end = min(count, int(round(event.end_s / hop_s)))
        if end <= begin:
            continue
        base = midi_to_hz(event.midi)
        span = np.arange(end - begin) * hop_s
        cents = _bend_cents(event.bend, span)
        f0[begin:end] = base * 2.0 ** (cents / 1200.0)
        voiced[begin:end] = True

    return Tune(hop_s=hop_s,
                f0_hz=[round(float(v), 4) for v in f0],
                voiced=[bool(v) for v in voiced],
                events=[asdict(e) for e in ordered],
                source=DIRECT, is_estimate=False, notes=notes)


def _bend_cents(bend: list, span: np.ndarray) -> np.ndarray:
    """Ornament detail inside one event, in cents, linearly interpolated.

    No smoothing and no quantisation: what the controls say is what the curve
    does.
    """
    if not bend:
        return np.zeros(len(span))
    points = sorted((float(t), float(c)) for t, c in bend)
    times = np.array([t for t, _ in points])
    values = np.array([c for _, c in points])
    return np.interp(span, times, values, left=values[0], right=values[-1])


def from_wav(path: Path, *, hop_s: float = HOP_SECONDS,
             fmin: float = 80.0, fmax: float = 1000.0,
             voiced_floor_db: float = -45.0) -> Tune:
    """Monophonic WAV to a curve. The adapter, and an estimate.

    Autocorrelation with a harmonic check, because a pitch estimate that can be
    out by an octave is not an estimate anyone should build on. Silence and
    unvoiced frames stay unvoiced; nothing is interpolated across them.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise TuneError(f"{path} holds no audio.")

    hop = max(1, int(round(rate * hop_s)))
    window = max(hop * 4, int(rate / fmin * 2))
    count = max(1, (len(audio) - window) // hop + 1)
    f0 = np.zeros(count)
    voiced = np.zeros(count, dtype=bool)
    floor = 10 ** (voiced_floor_db / 20.0)

    for index in range(count):
        frame = audio[index * hop:index * hop + window]
        if len(frame) < window or float(np.sqrt(np.mean(frame ** 2))) < floor:
            continue
        estimate = _harmonic_f0(frame - frame.mean(), rate, fmin, fmax)
        if estimate > 0:
            f0[index] = estimate
            voiced[index] = True

    return Tune(hop_s=hop_s,
                f0_hz=[round(float(v), 4) for v in f0],
                voiced=[bool(v) for v in voiced],
                events=[], source=WAV_ADAPTER, is_estimate=True,
                notes=(f"Estimated from {Path(path).name} by autocorrelation with "
                       f"a harmonic check. An estimate, not the melody's own "
                       f"controls; prefer direct controls where they exist."))


def _harmonic_f0(frame: np.ndarray, rate: int, fmin: float, fmax: float) -> float:
    """Autocorrelation, then a harmonic-sum check against octave errors.

    Autocorrelation halves pitches through period doubling and peak-picking
    doubles them when the second harmonic is louder. Both happened to me on real
    renders today, in opposite directions, so neither is trusted alone.
    """
    correlation = np.correlate(frame, frame, "full")[len(frame) - 1:]
    if correlation[0] <= 0:
        return 0.0
    low, high = int(rate / fmax), min(int(rate / fmin), len(correlation) - 1)
    if high <= low:
        return 0.0
    lag = int(np.argmax(correlation[low:high])) + low
    if correlation[lag] < 0.3 * correlation[0]:
        return 0.0
    candidate = rate / lag

    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=1 << 15))
    resolution = rate / (1 << 15)

    def score(hz: float) -> float:
        if not fmin <= hz <= fmax:
            return -1.0
        total = 0.0
        for harmonic in range(1, 6):
            index = int(round(hz * harmonic / resolution))
            if 1 < index < len(spectrum) - 1:
                total += float(np.max(spectrum[index - 1:index + 2])) / harmonic
        return total

    best = max((candidate, candidate / 2.0, candidate * 2.0), key=score)
    return float(best) if score(best) > 0 else 0.0
