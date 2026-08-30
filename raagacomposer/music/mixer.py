"""Mix engine (spec sections 12.24, 19).

Renders the arrangement's tracks, places them in the stereo field, applies
per-family treatment so the vocal keeps its space, and glues the result with
bus compression and a limiter.  Three products come out of the same path: the
full mix, the instrumental, and the vocal-only master.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..audio import dsp
from ..core.models import ArrangementVersion, MelodyVersion, Track
from .instruments import get as get_instrument
from .synth import render_track

# Per-family mix treatment: (high-pass Hz, low-pass Hz, reverb send, comp ratio)
FAMILY_TREATMENT: Dict[str, Tuple[float, float, float, float]] = {
    "percussion": (60.0, 16000.0, 0.10, 3.0),
    "bass": (35.0, 4000.0, 0.05, 3.5),
    "drone": (50.0, 6000.0, 0.18, 1.5),
    "plucked-string": (110.0, 14000.0, 0.22, 2.0),
    "struck-string": (140.0, 15000.0, 0.26, 2.0),
    "bowed-string": (90.0, 13000.0, 0.24, 2.2),
    "wind": (130.0, 13000.0, 0.22, 2.4),
    "keyboard": (80.0, 14000.0, 0.18, 2.2),
    "voice": (110.0, 12000.0, 0.28, 2.0),
    "synth": (60.0, 12000.0, 0.24, 1.8),
}

ROLE_GAIN = {
    "lead": 1.0, "counter": 0.7, "pad": 0.55, "bass": 0.85,
    "rhythm": 0.8, "fill": 0.75, "drone": 0.45,
}


@dataclass
class MixResult:
    audio: np.ndarray
    sample_rate: int
    duration: float
    loudness_db: float
    peak_db: float
    track_count: int
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.duration:.1f}s, {self.track_count} track(s), "
                f"loudness {self.loudness_db:.1f} dB, peak {self.peak_db:.1f} dBFS")


def audible_tracks(arrangement: ArrangementVersion) -> List[Track]:
    if arrangement is None:
        return []
    if arrangement.any_solo:
        return [t for t in arrangement.tracks if t.solo and not t.mute]
    return [t for t in arrangement.tracks if not t.mute]


def render_instrumental(arrangement: Optional[ArrangementVersion], sr: int,
                        total_seconds: float,
                        progress: Optional[Callable[[float, str], None]] = None,
                        cancelled: Optional[Callable[[], bool]] = None
                        ) -> Tuple[np.ndarray, int]:
    """Sum every audible track into a stereo bed."""
    bus = dsp.silence(total_seconds, sr, stereo=True)
    tracks = audible_tracks(arrangement) if arrangement else []
    for i, track in enumerate(tracks):
        if cancelled and cancelled():
            break
        if progress:
            progress(i / max(1, len(tracks)), f"Rendering {track.label}")
        inst = get_instrument(track.instrument)
        mono = render_track(track, sr, total_seconds)
        if not np.any(mono):
            continue
        family = inst.family if inst else "keyboard"
        hp, lp, send, ratio = FAMILY_TREATMENT.get(family, (80.0, 14000.0, 0.2, 2.0))
        mono = dsp.high_pass(mono, hp, sr)
        if lp < sr / 2:
            mono = dsp.low_pass(mono, lp, sr)
        mono = dsp.compressor(mono, -20.0, ratio, 0.015, 0.2, sr, makeup_db=1.0)
        stereo = dsp.pan_mono(mono, track.pan)
        if send > 0:
            stereo = dsp.reverb(stereo, sr, size=0.45, wet=send, damping=0.5)
        gain = ROLE_GAIN.get(track.role, 0.8)
        bus = dsp.mix_into(bus, dsp.pad_to(stereo, len(bus)), 0, gain)
    return bus.astype(np.float32), len(tracks)


def mix(arrangement: Optional[ArrangementVersion], vocal: Optional[np.ndarray],
        sr: int, total_seconds: float, kind: str = "full",
        vocal_gain: float = 1.0,
        progress: Optional[Callable[[float, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None) -> MixResult:
    notes: List[str] = []
    total_seconds = max(1.0, float(total_seconds))

    if kind == "vocal_only":
        bed = dsp.silence(total_seconds, sr)
        track_count = 0
    else:
        bed, track_count = render_instrumental(arrangement, sr, total_seconds,
                                               progress, cancelled)
        if track_count == 0:
            notes.append("No audible instrument tracks; the mix is vocal only.")

    if progress:
        progress(0.75, "Placing the vocal")

    out = dsp.pad_to(dsp.as_stereo(bed), int(total_seconds * sr))
    if vocal is not None and kind != "instrumental" and len(vocal):
        v = dsp.pad_to(dsp.as_stereo(vocal), len(out))
        if kind == "full" and track_count:
            # Duck the bed slightly under the voice so lyrics stay intelligible.
            key = dsp.as_mono(v)
            env = np.abs(key)
            win = max(1, int(0.05 * sr))
            env = np.convolve(env, np.ones(win, dtype=np.float32) / win,
                              mode="same").astype(np.float32)
            duck = 1.0 - 0.28 * np.clip(env / max(1e-6, float(env.max())), 0, 1)
            out *= duck[:, None]
        out = out + v * vocal_gain
    elif kind == "full":
        notes.append("No vocal render yet; the mix is instrumental.")

    if progress:
        progress(0.9, "Bus processing")

    out = dsp.high_pass(out, 28.0, sr)
    out = dsp.compressor(out, -16.0, 2.0, 0.03, 0.25, sr, makeup_db=1.5)
    out = dsp.peaking_eq(out, 2800.0, 1.2, 0.7, sr)
    out = dsp.shelf(out, 9000.0, 1.5, sr, "high")
    out = dsp.normalize_loudness(out, sr, -14.5)
    out = dsp.limiter(out, -0.8, sr)
    out = dsp.fade(out, 0.02, 0.4, sr)

    return MixResult(audio=out.astype(np.float32), sample_rate=sr,
                     duration=len(out) / sr,
                     loudness_db=dsp.loudness_db(out, sr),
                     peak_db=dsp.peak_db(out), track_count=track_count,
                     notes=notes)


def stems(arrangement: ArrangementVersion, sr: int, total_seconds: float
          ) -> Dict[str, np.ndarray]:
    """Per-track stereo stems for export."""
    out: Dict[str, np.ndarray] = {}
    for track in (arrangement.tracks if arrangement else []):
        mono = render_track(track, sr, total_seconds)
        if not np.any(mono):
            continue
        inst = get_instrument(track.instrument)
        family = inst.family if inst else "keyboard"
        hp, lp, send, ratio = FAMILY_TREATMENT.get(family, (80.0, 14000.0, 0.2, 2.0))
        mono = dsp.high_pass(mono, hp, sr)
        stereo = dsp.pan_mono(mono, track.pan)
        stereo = dsp.reverb(stereo, sr, 0.45, send)
        out[f"{track.label} ({track.role})"] = dsp.limiter(stereo, -1.0, sr)
    return out
