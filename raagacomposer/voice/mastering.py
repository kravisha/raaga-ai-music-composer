"""Vocal processing and mastering (spec section 4 step 7).

"Give me the song without instruments" must not return a dry demo.  The master
chain here is a full vocal production pass: cleanup, gain staging, corrective
and tonal EQ, compression, de-essing, ambience, stereo treatment, loudness
normalisation and limiting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..audio import dsp
from ..core.models import VocalDirection


@dataclass
class VocalChainSettings:
    high_pass: float = 85.0
    mud_freq: float = 320.0
    mud_gain: float = -2.5
    body_freq: float = 200.0
    body_gain: float = 1.0
    presence_freq: float = 4200.0
    presence_gain: float = 3.0
    air_freq: float = 11000.0
    air_gain: float = 2.5
    comp_threshold: float = -20.0
    comp_ratio: float = 3.2
    comp_attack: float = 0.012
    comp_release: float = 0.18
    deess_freq: float = 6800.0
    deess_amount: float = 0.65
    reverb_size: float = 0.45
    reverb_wet: float = 0.18
    delay_time: float = 0.3
    delay_wet: float = 0.1
    width: float = 1.2
    denoise: float = 0.25
    target_loudness: float = -14.0
    ceiling_db: float = -0.8


def settings_for(direction: VocalDirection, kind: str = "master"
                 ) -> VocalChainSettings:
    s = VocalChainSettings()
    style = (direction.style or "romantic").lower()
    if style in ("soft", "intimate", "smooth"):
        s.reverb_wet, s.reverb_size = 0.22, 0.4
        s.presence_gain, s.comp_ratio = 2.2, 3.6
        s.delay_wet = 0.07
    elif style in ("strong", "energetic", "dramatic"):
        s.reverb_wet, s.reverb_size = 0.13, 0.35
        s.presence_gain, s.comp_ratio = 3.8, 4.0
        s.comp_threshold = -22.0
    elif style in ("sad", "emotional"):
        s.reverb_wet, s.reverb_size = 0.26, 0.6
        s.delay_wet = 0.14
        s.presence_gain = 2.6
    elif style == "devotional":
        s.reverb_wet, s.reverb_size = 0.3, 0.7
        s.width = 1.35
    if kind != "master":
        # Preview: lighter touch and faster to render.
        s.reverb_wet *= 0.6
        s.delay_wet *= 0.4
        s.denoise = 0.0
        s.target_loudness = -18.0
    s.reverb_wet *= 0.6 + 0.8 * direction.sustain
    return s


def process_vocal(audio: np.ndarray, sr: int, direction: VocalDirection,
                  kind: str = "master",
                  settings: Optional[VocalChainSettings] = None) -> np.ndarray:
    """Run the vocal chain. ``kind`` is ``preview`` or ``master``."""
    s = settings or settings_for(direction, kind)
    x = dsp.as_stereo(audio)
    if len(x) == 0:
        return x

    if s.denoise > 0:
        x = dsp.denoise(x, sr, s.denoise)
    x = dsp.gate(x, -58.0, sr)
    x = dsp.high_pass(x, s.high_pass, sr, order=2)

    # Corrective then tonal EQ.
    x = dsp.peaking_eq(x, s.mud_freq, s.mud_gain, 1.1, sr)
    x = dsp.peaking_eq(x, s.body_freq, s.body_gain, 0.9, sr)
    x = dsp.peaking_eq(x, s.presence_freq, s.presence_gain, 0.8, sr)
    x = dsp.shelf(x, s.air_freq, s.air_gain, sr, "high")

    x = dsp.compressor(x, s.comp_threshold, s.comp_ratio, s.comp_attack,
                       s.comp_release, sr)
    x = dsp.de_esser(x, sr, s.deess_freq, -30.0, s.deess_amount)

    # A second, gentler pass evens out the long phrases.
    x = dsp.compressor(x, -14.0, 2.0, 0.05, 0.3, sr, makeup_db=1.0)

    if s.delay_wet > 0:
        x = dsp.delay(x, sr, s.delay_time, 0.25, s.delay_wet)
    if s.reverb_wet > 0:
        x = dsp.reverb(x, sr, s.reverb_size, s.reverb_wet, damping=0.45)
    if s.width != 1.0:
        x = dsp.widen(x, s.width)

    x = dsp.normalize_loudness(x, sr, s.target_loudness)
    x = dsp.limiter(x, s.ceiling_db, sr)
    return x.astype(np.float32)


def master_vocal_only(audio: np.ndarray, sr: int,
                      direction: VocalDirection) -> np.ndarray:
    """The studio-quality vocal-only master: no instruments, fully produced."""
    out = process_vocal(audio, sr, direction, kind="master")
    out = dsp.fade(out, 0.02, 0.35, sr)
    return out


def quick_preview(audio: np.ndarray, sr: int,
                  direction: VocalDirection) -> np.ndarray:
    return process_vocal(audio, sr, direction, kind="preview")


def report(audio: np.ndarray, sr: int) -> str:
    return (f"peak {dsp.peak_db(audio):.1f} dBFS | "
            f"rms {dsp.rms_db(audio):.1f} dB | "
            f"loudness {dsp.loudness_db(audio, sr):.1f} dB (K-weighted)")
