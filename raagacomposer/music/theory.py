"""Small pitch/time helpers shared by the melody, arrangement and synth code."""
from __future__ import annotations

from typing import List

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def freq_to_midi(freq: float) -> float:
    import math
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def beat_seconds(bpm: float) -> float:
    return 60.0 / max(1.0, float(bpm))


def cycle_seconds(bpm: float, beats_per_cycle: int) -> float:
    return beat_seconds(bpm) * max(1, int(beats_per_cycle))


def quantise(t: float, bpm: float, subdivision: float = 0.25) -> float:
    step = beat_seconds(bpm) * subdivision
    return round(t / step) * step


def fit_to_range(midi: int, low: int, high: int) -> int:
    """Transpose by octaves until the note sits inside [low, high]."""
    while midi < low:
        midi += 12
    while midi > high:
        midi -= 12
    return max(low, min(high, midi))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    return f"{int(m):d}:{s:05.2f}"


def format_time_short(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def spread(values: List[float], total: float) -> List[float]:
    """Scale a list of weights so it sums to *total*."""
    s = sum(values) or 1.0
    return [v * total / s for v in values]
