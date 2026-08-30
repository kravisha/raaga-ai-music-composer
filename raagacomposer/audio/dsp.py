"""Audio DSP primitives.

Everything the mix and vocal-mastering chains need: filters, dynamics, time
effects, stereo placement and loudness.  All functions take and return float32
numpy arrays, mono as ``(n,)`` and stereo as ``(n, 2)``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal

EPS = 1e-12


# --------------------------------------------------------------------------
# shape helpers
# --------------------------------------------------------------------------
def as_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


def as_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x if x.ndim == 1 else x.mean(axis=1)


def silence(seconds: float, sr: int, stereo: bool = True) -> np.ndarray:
    n = max(0, int(round(seconds * sr)))
    return np.zeros((n, 2), dtype=np.float32) if stereo else np.zeros(n, dtype=np.float32)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad_shape = (n - len(x),) + x.shape[1:]
    return np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=0)


def mix_into(dest: np.ndarray, src: np.ndarray, at_sample: int,
             gain: float = 1.0) -> np.ndarray:
    """Add *src* into *dest* at a sample offset, growing *dest* if needed."""
    if len(src) == 0:
        return dest
    at_sample = max(0, int(at_sample))
    end = at_sample + len(src)
    if end > len(dest):
        dest = pad_to(dest, end)
    if dest.ndim == 2 and src.ndim == 1:
        src = as_stereo(src)
    elif dest.ndim == 1 and src.ndim == 2:
        src = as_mono(src)
    dest[at_sample:end] += (src * gain).astype(dest.dtype)
    return dest


def fade(x: np.ndarray, fade_in: float, fade_out: float, sr: int) -> np.ndarray:
    x = x.copy()
    n_in = min(len(x), int(fade_in * sr))
    n_out = min(len(x) - n_in, int(fade_out * sr))
    if n_in > 0:
        ramp = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
        x[:n_in] *= ramp if x.ndim == 1 else ramp[:, None]
    if n_out > 0:
        ramp = np.linspace(1.0, 0.0, n_out, dtype=np.float32)
        x[-n_out:] *= ramp if x.ndim == 1 else ramp[:, None]
    return x


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def _sos_apply(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return signal.sosfilt(sos, x).astype(np.float32)
    return np.stack([signal.sosfilt(sos, x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def high_pass(x: np.ndarray, freq: float, sr: int, order: int = 2) -> np.ndarray:
    freq = max(10.0, min(freq, sr / 2 - 100))
    return _sos_apply(x, signal.butter(order, freq, "highpass", fs=sr, output="sos"))


def low_pass(x: np.ndarray, freq: float, sr: int, order: int = 2) -> np.ndarray:
    freq = max(50.0, min(freq, sr / 2 - 100))
    return _sos_apply(x, signal.butter(order, freq, "lowpass", fs=sr, output="sos"))


def band_pass(x: np.ndarray, low: float, high: float, sr: int,
              order: int = 2) -> np.ndarray:
    low = max(20.0, low)
    high = min(high, sr / 2 - 100)
    if high <= low:
        return x
    return _sos_apply(x, signal.butter(order, [low, high], "bandpass", fs=sr,
                                       output="sos"))


def peaking_eq(x: np.ndarray, freq: float, gain_db: float, q: float,
               sr: int) -> np.ndarray:
    """Single peaking biquad (RBJ cookbook)."""
    if abs(gain_db) < 1e-3:
        return x
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * max(20.0, min(freq, sr / 2 - 100)) / sr
    alpha = np.sin(w0) / (2 * max(0.1, q))
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    sos = signal.tf2sos(b, a)
    return _sos_apply(x, sos)


def shelf(x: np.ndarray, freq: float, gain_db: float, sr: int,
          kind: str = "high") -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return x
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * max(20.0, min(freq, sr / 2 - 100)) / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / 0.9 - 1) + 2)
    sq = 2 * np.sqrt(A) * alpha
    if kind == "high":
        b = [A * ((A + 1) + (A - 1) * cos_w0 + sq),
             -2 * A * ((A - 1) + (A + 1) * cos_w0),
             A * ((A + 1) + (A - 1) * cos_w0 - sq)]
        a = [(A + 1) - (A - 1) * cos_w0 + sq,
             2 * ((A - 1) - (A + 1) * cos_w0),
             (A + 1) - (A - 1) * cos_w0 - sq]
    else:
        b = [A * ((A + 1) - (A - 1) * cos_w0 + sq),
             2 * A * ((A - 1) - (A + 1) * cos_w0),
             A * ((A + 1) - (A - 1) * cos_w0 - sq)]
        a = [(A + 1) + (A - 1) * cos_w0 + sq,
             -2 * ((A - 1) + (A + 1) * cos_w0),
             (A + 1) + (A - 1) * cos_w0 - sq]
    return _sos_apply(x, signal.tf2sos(b, a))


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------
def _envelope(x: np.ndarray, attack: float, release: float, sr: int) -> np.ndarray:
    mono = np.abs(as_mono(x)).astype(np.float32)
    a = float(np.exp(-1.0 / max(1.0, attack * sr)))
    r = float(np.exp(-1.0 / max(1.0, release * sr)))
    # One-pole attack/release follower, vectorised per block for speed.
    env = np.empty_like(mono)
    y = 0.0
    for i in range(len(mono)):
        v = mono[i]
        coeff = a if v > y else r
        y = coeff * y + (1.0 - coeff) * v
        env[i] = y
    return env


def _envelope_fast(x: np.ndarray, attack: float, release: float,
                   sr: int) -> np.ndarray:
    """Vectorised follower: peak-hold decay, close enough for musical use."""
    mono = np.abs(as_mono(x)).astype(np.float32)
    win = max(1, int(attack * sr))
    if win > 1:
        kernel = np.ones(win, dtype=np.float32) / win
        mono = np.convolve(mono, kernel, mode="same").astype(np.float32)
    r = float(np.exp(-1.0 / max(1.0, release * sr)))
    out = signal.lfilter([1 - r], [1, -r], mono).astype(np.float32)
    return np.maximum(out, mono * 0.35)


def compressor(x: np.ndarray, threshold_db: float = -18.0, ratio: float = 3.0,
               attack: float = 0.01, release: float = 0.15, sr: int = 44100,
               makeup_db: Optional[float] = None, knee_db: float = 6.0
               ) -> np.ndarray:
    env = _envelope_fast(x, attack, release, sr)
    env_db = 20 * np.log10(np.maximum(env, EPS))
    over = env_db - threshold_db
    # Soft knee.
    gain_db = np.zeros_like(over)
    knee = max(0.1, knee_db)
    below = over < -knee / 2
    above = over > knee / 2
    middle = ~below & ~above
    gain_db[above] = -(over[above] - over[above] / ratio)
    t = (over[middle] + knee / 2) / knee
    gain_db[middle] = -((1 - 1 / ratio) * (over[middle] + knee / 2) * t / 2)
    gain = (10 ** (gain_db / 20.0)).astype(np.float32)
    out = x * (gain if x.ndim == 1 else gain[:, None])
    if makeup_db is None:
        makeup_db = max(0.0, -threshold_db * (1 - 1 / ratio) * 0.5)
    return (out * (10 ** (makeup_db / 20.0))).astype(np.float32)


def _zero_phase_high_pass(x: np.ndarray, freq: float, sr: int,
                          order: int = 2) -> np.ndarray:
    """High-pass with no phase shift, so the band can safely be subtracted."""
    freq = max(10.0, min(freq, sr / 2 - 100))
    sos = signal.butter(order, freq, "highpass", fs=sr, output="sos")
    pad = min(max(0, len(x) - 1), 3 * (2 * order + 1))
    if x.ndim == 1:
        return signal.sosfiltfilt(sos, x, padlen=pad).astype(np.float32)
    return np.stack([signal.sosfiltfilt(sos, x[:, c], padlen=pad)
                     for c in range(x.shape[1])], axis=1).astype(np.float32)


def de_esser(x: np.ndarray, sr: int, freq: float = 6500.0,
             threshold_db: float = -28.0, amount: float = 0.7) -> np.ndarray:
    """Duck the sibilant band only, leaving the rest of the voice alone.

    The band is extracted with a zero-phase filter and only the *excess* is
    subtracted, so with no reduction the signal is returned untouched and a
    phase-rotated band can never add energy back at the sibilant frequency.
    """
    sibilant = _zero_phase_high_pass(x, freq, sr, order=2)
    env = _envelope_fast(sibilant, 0.002, 0.05, sr)
    env_db = 20 * np.log10(np.maximum(env, EPS))
    reduce_db = np.minimum(0.0, -(env_db - threshold_db)) * amount
    gain = (10 ** (reduce_db / 20.0)).astype(np.float32)
    excess = sibilant * (1.0 - (gain if x.ndim == 1 else gain[:, None]))
    return (x - excess).astype(np.float32)


def limiter(x: np.ndarray, ceiling_db: float = -0.6, sr: int = 44100,
            lookahead: float = 0.005) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20.0)
    peak = np.abs(as_mono(x))
    n = max(1, int(lookahead * sr))
    if n > 1:
        peak = np.maximum(peak, np.convolve(peak, np.ones(n, dtype=np.float32),
                                            mode="same") / n)
    over = np.maximum(peak / ceiling, 1.0)
    # Smooth the gain reduction so it does not distort.
    win = max(1, int(0.003 * sr))
    kernel = np.ones(win, dtype=np.float32) / win
    over = np.convolve(over, kernel, mode="same").astype(np.float32)
    gain = 1.0 / np.maximum(over, 1.0)
    out = x * (gain if x.ndim == 1 else gain[:, None])
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


def gate(x: np.ndarray, threshold_db: float = -55.0, sr: int = 44100,
         attack: float = 0.005, release: float = 0.08) -> np.ndarray:
    env = _envelope_fast(x, attack, release, sr)
    thr = 10 ** (threshold_db / 20.0)
    gain = np.clip((env / max(thr, EPS)), 0.0, 1.0).astype(np.float32)
    return (x * (gain if x.ndim == 1 else gain[:, None])).astype(np.float32)


# --------------------------------------------------------------------------
# time effects
# --------------------------------------------------------------------------
def _impulse_response(seconds: float, sr: int, decay: float, damping: float,
                      seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal((n, 2)).astype(np.float32)
    env = np.exp(-decay * t).astype(np.float32)[:, None]
    ir = noise * env
    # Early reflections give the tail a room rather than a wash.
    for delay_s, g in ((0.011, 0.5), (0.019, 0.35), (0.029, 0.28), (0.041, 0.2)):
        d = int(delay_s * sr)
        if d < n:
            ir[d:] += ir[:n - d] * g
    ir = low_pass(ir, max(1200.0, 12000.0 * (1.0 - damping)), sr, order=2)
    ir /= max(EPS, np.abs(ir).max())
    return ir.astype(np.float32)


_IR_CACHE: dict = {}


def reverb(x: np.ndarray, sr: int, size: float = 0.5, wet: float = 0.2,
           damping: float = 0.4, predelay: float = 0.02) -> np.ndarray:
    if wet <= 0.001:
        return x
    seconds = 0.5 + 2.6 * max(0.0, min(1.0, size))
    decay = 6.0 / max(0.3, seconds)
    key = (round(seconds, 2), sr, round(decay, 2), round(damping, 2))
    ir = _IR_CACHE.get(key)
    if ir is None:
        ir = _impulse_response(seconds, sr, decay, damping)
        _IR_CACHE[key] = ir
    st = as_stereo(x)
    pre = int(predelay * sr)
    wet_sig = np.zeros_like(st)
    for c in range(2):
        conv = signal.fftconvolve(st[:, c], ir[:, c])[:len(st)]
        wet_sig[:len(conv), c] = conv
    if pre:
        wet_sig = np.concatenate([np.zeros((pre, 2), np.float32), wet_sig])[:len(st)]
    peak = np.abs(wet_sig).max()
    if peak > EPS:
        wet_sig = wet_sig / peak * max(EPS, np.abs(st).max())
    out = (1.0 - wet * 0.5) * st + wet * wet_sig
    return out.astype(np.float32)


def delay(x: np.ndarray, sr: int, time: float = 0.28, feedback: float = 0.28,
          wet: float = 0.16, spread: float = 0.4) -> np.ndarray:
    if wet <= 0.001:
        return x
    st = as_stereo(x).copy()
    out = st.copy()
    for c in range(2):
        t = time * (1.0 + (spread * 0.25 if c else -spread * 0.25))
        d = max(1, int(t * sr))
        buf = np.zeros(len(st) + d * 6, dtype=np.float32)
        buf[:len(st)] = st[:, c]
        tap = np.zeros_like(buf)
        g = 1.0
        for k in range(1, 7):
            g *= feedback
            if g < 0.01:
                break
            tap[d * k:d * k + len(st)] += st[:, c] * g
        out[:, c] = st[:, c] * (1 - wet * 0.4) + tap[:len(st)] * wet
    return out.astype(np.float32)


def chorus(x: np.ndarray, sr: int, depth_ms: float = 6.0, rate: float = 0.5,
           wet: float = 0.25) -> np.ndarray:
    st = as_stereo(x)
    n = len(st)
    t = np.arange(n) / sr
    out = st.copy()
    for c in range(2):
        mod = (depth_ms / 1000.0) * sr * (0.5 + 0.5 * np.sin(
            2 * np.pi * rate * t + (0.0 if c == 0 else np.pi / 2)))
        idx = np.clip(np.arange(n) - mod, 0, n - 1)
        voiced = np.interp(idx, np.arange(n), st[:, c]).astype(np.float32)
        out[:, c] = (1 - wet) * st[:, c] + wet * voiced
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# stereo and level
# --------------------------------------------------------------------------
def pan_mono(x: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan of a mono signal, -1 left .. +1 right."""
    mono = as_mono(x)
    p = (max(-1.0, min(1.0, pan)) + 1.0) / 2.0
    left = np.cos(p * np.pi / 2)
    right = np.sin(p * np.pi / 2)
    return np.stack([mono * left, mono * right], axis=1).astype(np.float32)


def widen(x: np.ndarray, width: float = 1.3) -> np.ndarray:
    st = as_stereo(x)
    mid = (st[:, 0] + st[:, 1]) * 0.5
    side = (st[:, 0] - st[:, 1]) * 0.5 * width
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    mono = as_mono(x)
    if len(mono) == 0:
        return -120.0
    return float(20 * np.log10(max(EPS, np.sqrt(np.mean(mono ** 2)))))


def peak_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    return float(20 * np.log10(max(EPS, float(np.abs(x).max()))))


def loudness_db(x: np.ndarray, sr: int) -> float:
    """K-weighted RMS: a workable stand-in for LUFS without extra deps."""
    weighted = shelf(as_mono(x), 1500.0, 4.0, sr, "high")
    weighted = high_pass(weighted, 60.0, sr)
    return rms_db(weighted)


def normalize_loudness(x: np.ndarray, sr: int, target_db: float = -16.0,
                       max_gain_db: float = 24.0) -> np.ndarray:
    current = loudness_db(x, sr)
    if current <= -119:
        return x
    gain_db = max(-max_gain_db, min(max_gain_db, target_db - current))
    return (x * (10 ** (gain_db / 20.0))).astype(np.float32)


def normalize_peak(x: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= EPS:
        return x
    target = 10 ** (target_db / 20.0)
    return (x * (target / peak)).astype(np.float32)


def denoise(x: np.ndarray, sr: int, amount: float = 0.4) -> np.ndarray:
    """Gentle spectral-floor reduction; enough to clean a synthetic take."""
    if amount <= 0.001:
        return x
    st = as_stereo(x)
    out = np.zeros_like(st)
    nper = 1024
    for c in range(2):
        f, t, Z = signal.stft(st[:, c], fs=sr, nperseg=nper, noverlap=nper // 2)
        mag = np.abs(Z)
        floor = np.percentile(mag, 8, axis=1, keepdims=True)
        reduced = np.maximum(mag - floor * amount * 2.0, mag * (1 - amount))
        Z2 = Z * (reduced / np.maximum(mag, EPS))
        _, rec = signal.istft(Z2, fs=sr, nperseg=nper, noverlap=nper // 2)
        out[:min(len(rec), len(st)), c] = rec[:len(st)]
    return out.astype(np.float32)


def soft_clip(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Saturate toward full scale. Unity for a full-scale input, never above it."""
    shaped = np.tanh(x * drive) / max(EPS, np.tanh(drive))
    return np.clip(shaped, -1.0, 1.0).astype(np.float32)
