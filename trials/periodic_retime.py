"""Phase-synchronous period tiling — retime without moving the pitch.

The previous implementation advanced its read pointer by a whole period while
advancing its write pointer by `period - fade`. Every repeat therefore landed a
cycle at the wrong phase, and the overlap shifted the pitch: a 200 Hz sine came
back at 266.67 Hz, +498 cents, reproduced independently. A guide whose notes are
half a fifth out fails the score contract whatever a recogniser makes of the
words.

This is the ordinary PSOLA construction instead. Grains are two periods long and
Hann-windowed, and they are written at output positions spaced **exactly one
period apart**. Output grain spacing is what sets the pitch, so keeping it equal
to the input period keeps the pitch. Which grain is read is chosen by mapping
the output position back through the time-scale factor and snapping to the
nearest whole period, so grains are always phase-aligned.

Length is exact because the output is allocated at the requested size and
truncated to it.
"""
from __future__ import annotations

import numpy as np


def retime(block: np.ndarray, target: int, period: int) -> np.ndarray:
    """`block` at a new length, same pitch, exactly `target` samples."""
    if target <= 0:
        return np.zeros(0)
    block = np.asarray(block, dtype=np.float64)
    if len(block) == 0:
        return np.zeros(target)
    period = int(max(16, period))

    # Too short to hold two periods: repeat it end to end, phase-aligned, so
    # the pitch is set by the repeat spacing rather than by resampling. The old
    # code interpolated here, which changes pitch by construction.
    if len(block) < 2 * period:
        usable = (len(block) // period) * period
        if usable < period:
            padded = np.zeros(period)
            padded[:len(block)] = block[:period]
            block, usable = padded, period
        else:
            block = block[:usable]

    grain = 2 * period
    window = np.hanning(grain)
    out = np.zeros(target + grain)
    weight = np.zeros(target + grain)

    marks = np.arange(0, max(len(block) - grain, 0) + 1, period)
    if len(marks) == 0:
        marks = np.array([0])
    scale = (len(block) - grain) / max(target - 1, 1) if target > 1 else 0.0

    position = 0
    while position < target:
        # Where in the source this output moment corresponds to, snapped to a
        # whole period so the grain is in phase with its neighbours.
        source = position * scale
        mark = int(round(source / period)) * period
        mark = int(np.clip(mark, 0, max(len(block) - grain, 0)))
        piece = block[mark:mark + grain]
        if len(piece) < grain:
            piece = np.pad(piece, (0, grain - len(piece)))
        out[position:position + grain] += piece * window
        weight[position:position + grain] += window
        position += period            # output hop == period, so pitch holds

    busy = weight > 1e-9
    out[busy] /= weight[busy]
    return out[:target]


def dominant_hz(x: np.ndarray, rate: int) -> float:
    """Strongest spectral component, for the invariant check."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=1 << 16))
    return float(np.fft.rfftfreq(1 << 16, 1 / rate)[int(np.argmax(spectrum))])
