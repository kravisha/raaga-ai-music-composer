"""Offline instrument synthesis.

A local synthesis engine keeps the whole workflow runnable with no cloud
credentials: every instrument in the catalog can be heard immediately, and a
cloud instrument provider can replace this later behind the same interface
(:mod:`raagacomposer.providers`).

Notes are rendered by additive synthesis over an instantaneous-frequency curve,
so vibrato, gamaka (kampita oscillation and slides) and legato glide are all
just modulation of that curve rather than post-processing.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.signal import lfilter

from ..core.models import Note, Region, Track
from .instruments import Instrument, get as get_instrument
from .theory import midi_to_freq

_NOISE_CACHE: Dict[int, np.ndarray] = {}


def _noise(n: int, seed: int = 0) -> np.ndarray:
    key = (n, seed).__hash__()
    cached = _NOISE_CACHE.get(key)
    if cached is None or len(cached) < n:
        rng = np.random.default_rng(seed or 12345)
        cached = rng.standard_normal(max(n, 4096)).astype(np.float32)
        _NOISE_CACHE[key] = cached
    return cached[:n]


def _adsr(n: int, sr: int, inst: Instrument, sustain_len: int) -> np.ndarray:
    a = max(1, int(inst.attack * sr))
    d = max(1, int(inst.decay * sr))
    r = max(1, int(inst.release * sr))
    env = np.zeros(n, dtype=np.float32)
    a = min(a, n)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    d_end = min(n, a + d)
    if d_end > a:
        env[a:d_end] = np.linspace(1.0, inst.sustain, d_end - a, dtype=np.float32)
    s_end = min(n, max(d_end, sustain_len))
    if s_end > d_end:
        env[d_end:s_end] = inst.sustain
    if n > s_end:
        tail = n - s_end
        env[s_end:] = np.linspace(inst.sustain, 0.0, tail, dtype=np.float32) ** 1.5
    return env


def _pluck_env(n: int, sr: int, inst: Instrument) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    attack = max(1, int(inst.attack * sr))
    env = np.exp(-inst.pluck_decay * t).astype(np.float32)
    env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    return env


def _freq_curve(note: Note, inst: Instrument, sr: int, n: int,
                prev_freq: Optional[float], gamaka_amount: float,
                rng: np.random.Generator) -> np.ndarray:
    base = midi_to_freq(note.midi)
    t = np.arange(n, dtype=np.float32) / sr
    ratio = np.ones(n, dtype=np.float32)

    # Vibrato, delayed so short notes stay clean.
    if inst.vibrato_depth > 0:
        onset = min(n, int(0.18 * n))
        depth = inst.vibrato_depth
        vib = np.sin(2 * np.pi * inst.vibrato_rate * t).astype(np.float32)
        ramp = np.ones(n, dtype=np.float32)
        if onset > 0:
            ramp[:onset] = np.linspace(0.0, 1.0, onset, dtype=np.float32)
        ratio *= 2 ** (depth * vib * ramp / 12.0)

    # Gamaka.
    g = (note.gamaka or "").lower()
    if g and gamaka_amount > 0:
        if g in ("kampita", "kampitam", "oscillate"):
            rate = 5.5 if note.duration > 0.4 else 7.0
            depth = 0.55 * gamaka_amount
            shape = np.sin(2 * np.pi * rate * t).astype(np.float32)
            hold = np.clip(t / max(0.05, note.duration * 0.25), 0, 1).astype(np.float32)
            ratio *= 2 ** (depth * shape * hold / 12.0)
        elif g in ("slide_up", "meend_up", "jaru_up"):
            span = max(1, int(n * 0.3))
            slide = np.ones(n, dtype=np.float32)
            slide[:span] = np.linspace(2 ** (-2.0 / 12), 1.0, span, dtype=np.float32)
            ratio *= slide
        elif g in ("slide_down", "meend_down", "jaru_down"):
            span = max(1, int(n * 0.3))
            slide = np.ones(n, dtype=np.float32)
            slide[:span] = np.linspace(2 ** (2.0 / 12), 1.0, span, dtype=np.float32)
            ratio *= slide

    # Legato glide from the previous note.
    if prev_freq and inst.glide > 0:
        span = max(1, int(min(0.14, inst.glide * 0.2) * sr))
        span = min(span, n)
        start_ratio = float(prev_freq) / max(1.0, base)
        if 0.4 < start_ratio < 2.5:
            glide = np.ones(n, dtype=np.float32)
            glide[:span] = np.linspace(start_ratio, 1.0, span, dtype=np.float32)
            ratio *= glide

    # A little human drift.
    drift = 1.0 + 0.0016 * np.sin(2 * np.pi * 0.7 * t + rng.random() * 6.28)
    return (base * ratio * drift).astype(np.float32)


def render_note(note: Note, inst: Instrument, sr: int,
                prev_freq: Optional[float] = None, gamaka_amount: float = 1.0,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng((seed * 7919 + int(note.start * 1000)) % (2 ** 31))
    tail = inst.release + (0.6 if inst.pluck else 0.0)
    total = max(0.05, note.duration + tail)
    n = int(total * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    freq = _freq_curve(note, inst, sr, n, prev_freq, gamaka_amount, rng)
    phase = 2 * np.pi * np.cumsum(freq) / sr

    voices = max(1, inst.voices)
    out = np.zeros(n, dtype=np.float32)
    for v in range(voices):
        detune = 0.0 if voices == 1 else (v - (voices - 1) / 2) * inst.detune
        vphase = phase * (2 ** (detune / 12.0))
        sig = np.zeros(n, dtype=np.float32)
        for k, amp in enumerate(inst.harmonics, start=1):
            if amp <= 0.001:
                continue
            stretch = 1.0 + inst.inharmonicity * (k ** 2)
            roll = amp * (inst.brightness ** (1.0 if k <= 2 else 0.7))
            if k * float(freq.max()) * stretch > sr * 0.48:
                break
            sig += (roll * np.sin(vphase * k * stretch)).astype(np.float32)
        out += sig / voices
    peak = float(np.abs(out).max())
    if peak > 0:
        out /= peak

    if inst.pluck:
        env = _pluck_env(n, sr, inst)
    else:
        env = _adsr(n, sr, inst, int(note.duration * sr))
    out *= env

    if inst.noise > 0:
        nz = _noise(n, seed + note.midi) * inst.noise
        # One-pole tilt toward the instrument's breath/bow noise band.
        alpha = float(np.exp(-2 * np.pi * min(inst.noise_color, sr * 0.45) / sr))
        low = lfilter([1 - alpha], [1, -alpha], nz).astype(np.float32)
        out += (nz - low) * env * 0.6

    velocity = max(0.05, min(1.2, note.velocity / 100.0))
    return (out * velocity).astype(np.float32)


def render_notes(notes: Sequence[Note], inst: Instrument, sr: int,
                 total_seconds: Optional[float] = None, offset: float = 0.0,
                 gain: float = 1.0, gamaka_amount: float = 1.0,
                 seed: int = 0) -> np.ndarray:
    notes = sorted(notes, key=lambda n: n.start)
    if not notes:
        length = int((total_seconds or 0.0) * sr)
        return np.zeros(max(0, length), dtype=np.float32)
    if inst.percussive:
        return render_percussion(notes, inst, sr, total_seconds, offset, gain, seed)

    end = total_seconds if total_seconds is not None else \
        max(n.end for n in notes) - offset + inst.release + 1.0
    buf = np.zeros(max(1, int(end * sr) + sr), dtype=np.float32)
    prev_freq: Optional[float] = None
    prev_end = -10.0
    for note in notes:
        legato = (note.start - prev_end) < 0.08
        sig = render_note(note, inst, sr, prev_freq if legato else None,
                          gamaka_amount, seed)
        at = int((note.start - offset) * sr)
        if at < 0:
            sig = sig[-at:]
            at = 0
        if at >= len(buf):
            continue
        span = min(len(sig), len(buf) - at)
        buf[at:at + span] += sig[:span]
        prev_freq = midi_to_freq(note.midi)
        prev_end = note.end
    if total_seconds is not None:
        buf = buf[:max(1, int(total_seconds * sr))]
    return (buf * gain).astype(np.float32)


def render_percussion(notes: Sequence[Note], inst: Instrument, sr: int,
                      total_seconds: Optional[float] = None, offset: float = 0.0,
                      gain: float = 1.0, seed: int = 0) -> np.ndarray:
    end = total_seconds if total_seconds is not None else \
        (max((n.end for n in notes), default=0.0) - offset + 1.0)
    buf = np.zeros(max(1, int(end * sr) + sr), dtype=np.float32)
    rng = np.random.default_rng(seed or 4242)
    for note in notes:
        # The note's pitch selects which stroke of the drum is used.
        idx = int(note.midi) % max(1, len(inst.hit_freqs))
        f0 = inst.hit_freqs[idx]
        dec = inst.hit_decays[min(idx, len(inst.hit_decays) - 1)]
        length = int(min(1.2, dec * 4 + 0.05) * sr)
        t = np.arange(length, dtype=np.float32) / sr
        env = np.exp(-t / max(0.01, dec)).astype(np.float32)
        hit = np.zeros(length, dtype=np.float32)
        for mult, amp in ((1.0, 1.0), (1.58, 0.45), (2.24, 0.22)):
            hit += (amp * np.sin(2 * np.pi * f0 * mult * t)).astype(np.float32)
        nz = rng.standard_normal(length).astype(np.float32)
        nz_env = np.exp(-t / max(0.004, dec * 0.25)).astype(np.float32)
        hit = hit * env + nz * nz_env * inst.noise * 1.4
        hit *= max(0.1, note.velocity / 110.0)
        at = int((note.start - offset) * sr)
        if at < 0 or at >= len(buf):
            continue
        span = min(len(hit), len(buf) - at)
        buf[at:at + span] += hit[:span]
    if total_seconds is not None:
        buf = buf[:max(1, int(total_seconds * sr))]
    peak = float(np.abs(buf).max())
    if peak > 1.0:
        buf /= peak
    return (buf * gain).astype(np.float32)


def render_region(region: Region, instrument_key: str, sr: int,
                  total_seconds: Optional[float] = None) -> np.ndarray:
    inst = get_instrument(instrument_key)
    if inst is None:
        return np.zeros(int((total_seconds or 0.0) * sr), dtype=np.float32)
    return render_notes(region.notes, inst, sr, total_seconds=total_seconds,
                        gain=region.gain, seed=region.seed)


def render_track(track: Track, sr: int, total_seconds: float) -> np.ndarray:
    """Mono render of every region on a track, positioned on the timeline."""
    inst = get_instrument(track.instrument)
    buf = np.zeros(max(1, int(total_seconds * sr)), dtype=np.float32)
    if inst is None:
        return buf
    for region in track.regions:
        notes = region.notes
        if not notes:
            continue
        sig = render_notes(notes, inst, sr, total_seconds=None, offset=0.0,
                           gain=region.gain, seed=region.seed)
        span = min(len(sig), len(buf))
        buf[:span] += sig[:span]
    return (buf * track.gain).astype(np.float32)
