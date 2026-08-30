"""Unit tests: DSP primitives and the vocal production chain."""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.audio import dsp
from raagacomposer.core.models import VocalDirection
from raagacomposer.voice import mastering

pytestmark = pytest.mark.unit

SR = 22050


def tone(freq: float, seconds: float = 1.0, amp: float = 0.3,
         sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_energy(x: np.ndarray, low: float, high: float, sr: int = SR) -> float:
    mono = dsp.as_mono(x)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    mask = (freqs >= low) & (freqs <= high)
    return float(spec[mask].sum())


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def test_channel_conversions():
    mono = tone(440)
    stereo = dsp.as_stereo(mono)
    assert stereo.shape == (len(mono), 2)
    assert dsp.as_mono(stereo).shape == (len(mono),)
    assert dsp.as_stereo(stereo).shape == stereo.shape


def test_silence_and_padding():
    assert dsp.silence(0.5, SR).shape == (int(0.5 * SR), 2)
    assert dsp.silence(0.5, SR, stereo=False).shape == (int(0.5 * SR),)
    short = tone(440, 0.2)
    assert len(dsp.pad_to(short, SR)) == SR
    assert len(dsp.pad_to(short, 100)) == 100


def test_mix_into_grows_the_buffer_and_sums():
    dest = dsp.silence(1.0, SR)
    src = dsp.as_stereo(tone(440, 1.0))
    out = dsp.mix_into(dest, src, int(0.5 * SR))
    assert len(out) >= int(1.5 * SR)
    assert np.abs(out[int(0.6 * SR)]).max() > 0.0


def test_fade_ramps_both_ends():
    x = dsp.as_stereo(tone(440, 1.0))
    faded = dsp.fade(x, 0.1, 0.1, SR)
    assert np.abs(faded[0]).max() < 1e-6
    assert np.abs(faded[-1]).max() < 1e-3
    assert np.abs(faded[len(faded) // 2]).max() == pytest.approx(
        np.abs(x[len(x) // 2]).max(), rel=1e-3)


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def test_high_pass_removes_low_frequencies():
    mixed = tone(80) + tone(4000)
    out = dsp.high_pass(mixed, 1000, SR, order=4)
    assert band_energy(out, 40, 200) < band_energy(mixed, 40, 200) * 0.3
    assert band_energy(out, 3500, 4500) > band_energy(mixed, 3500, 4500) * 0.5


def test_low_pass_removes_high_frequencies():
    mixed = tone(120) + tone(6000)
    out = dsp.low_pass(mixed, 800, SR, order=4)
    assert band_energy(out, 5500, 6500) < band_energy(mixed, 5500, 6500) * 0.3


def test_band_pass_keeps_the_middle():
    mixed = tone(80) + tone(1000) + tone(8000)
    out = dsp.band_pass(mixed, 600, 1600, SR, order=4)
    assert band_energy(out, 900, 1100) > band_energy(out, 40, 200)
    assert band_energy(out, 900, 1100) > band_energy(out, 7000, 9000)


def test_peaking_eq_lifts_and_cuts():
    x = tone(1000)
    lifted = dsp.peaking_eq(x, 1000, 9.0, 1.0, SR)
    cut = dsp.peaking_eq(x, 1000, -9.0, 1.0, SR)
    assert band_energy(lifted, 900, 1100) > band_energy(x, 900, 1100)
    assert band_energy(cut, 900, 1100) < band_energy(x, 900, 1100)
    assert np.allclose(dsp.peaking_eq(x, 1000, 0.0, 1.0, SR), x)


def test_shelf_filters_move_their_band():
    x = tone(200) + tone(8000)
    high = dsp.shelf(x, 4000, 8.0, SR, "high")
    low = dsp.shelf(x, 400, 8.0, SR, "low")
    assert band_energy(high, 7000, 9000) > band_energy(x, 7000, 9000)
    assert band_energy(low, 150, 250) > band_energy(x, 150, 250)


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------
def test_compressor_reduces_the_dynamic_range():
    quiet = tone(440, 0.5, amp=0.05)
    loud = tone(440, 0.5, amp=0.9)
    signal = np.concatenate([quiet, loud])
    out = dsp.compressor(signal, threshold_db=-24.0, ratio=6.0, sr=SR,
                         makeup_db=0.0)
    before = np.abs(loud).max() / np.abs(quiet).max()
    after = np.abs(out[len(quiet):]).max() / max(1e-9,
                                                 np.abs(out[:len(quiet)]).max())
    assert after < before


def test_limiter_holds_the_ceiling():
    hot = tone(440, 1.0, amp=2.5)
    out = dsp.limiter(hot, ceiling_db=-1.0, sr=SR)
    assert dsp.peak_db(out) <= -0.9


def test_gate_silences_the_noise_floor():
    signal = np.concatenate([tone(440, 0.3, amp=0.0005), tone(440, 0.3, amp=0.5)])
    out = dsp.gate(signal, threshold_db=-45.0, sr=SR)
    assert np.abs(out[:int(0.2 * SR)]).max() < np.abs(signal[:int(0.2 * SR)]).max()
    assert np.abs(out[int(0.35 * SR):]).max() > 0.1


def test_de_esser_ducks_sibilance_only():
    voice = tone(300, 1.0, amp=0.4)
    sibilance = tone(7500, 1.0, amp=0.4)
    out = dsp.de_esser(voice + sibilance, SR, freq=6000, threshold_db=-40.0)
    assert band_energy(out, 7000, 8000) < band_energy(voice + sibilance, 7000, 8000)
    assert band_energy(out, 250, 350) == pytest.approx(
        band_energy(voice + sibilance, 250, 350), rel=0.25)


# --------------------------------------------------------------------------
# time effects and stereo
# --------------------------------------------------------------------------
def test_reverb_adds_a_tail_and_keeps_the_length():
    x = dsp.as_stereo(np.concatenate([tone(440, 0.1), np.zeros(SR, np.float32)]))
    out = dsp.reverb(x, SR, size=0.6, wet=0.5)
    assert out.shape == x.shape
    assert np.abs(out[int(0.4 * SR):]).max() > np.abs(x[int(0.4 * SR):]).max()


def test_reverb_is_a_no_op_when_dry():
    x = dsp.as_stereo(tone(440, 0.3))
    assert np.allclose(dsp.reverb(x, SR, wet=0.0), x)


def test_delay_produces_repeats():
    x = dsp.as_stereo(np.concatenate([tone(440, 0.05), np.zeros(SR, np.float32)]))
    out = dsp.delay(x, SR, time=0.2, feedback=0.5, wet=0.8)
    assert np.abs(out[int(0.18 * SR):int(0.28 * SR)]).max() > 0.001


def test_pan_is_constant_power():
    mono = tone(440, 0.5)
    left = dsp.pan_mono(mono, -1.0)
    right = dsp.pan_mono(mono, 1.0)
    centre = dsp.pan_mono(mono, 0.0)
    assert np.abs(left[:, 0]).max() > np.abs(left[:, 1]).max()
    assert np.abs(right[:, 1]).max() > np.abs(right[:, 0]).max()
    assert np.abs(centre[:, 0]).max() == pytest.approx(
        np.abs(centre[:, 1]).max(), rel=1e-6)


def test_widen_increases_the_side_signal():
    x = np.stack([tone(440), tone(445)], axis=1)
    wide = dsp.widen(x, 1.8)
    side_before = np.abs(x[:, 0] - x[:, 1]).max()
    side_after = np.abs(wide[:, 0] - wide[:, 1]).max()
    assert side_after > side_before


# --------------------------------------------------------------------------
# metering
# --------------------------------------------------------------------------
def test_meters_report_sensible_values():
    x = tone(1000, 1.0, amp=0.5)
    assert dsp.peak_db(x) == pytest.approx(-6.0, abs=0.5)
    assert dsp.rms_db(x) < dsp.peak_db(x)
    assert dsp.loudness_db(x, SR) > -60
    assert dsp.rms_db(np.zeros(10, np.float32)) < -100


def test_normalisation_hits_its_targets():
    x = tone(1000, 1.0, amp=0.02)
    louder = dsp.normalize_loudness(x, SR, target_db=-16.0)
    assert dsp.loudness_db(louder, SR) == pytest.approx(-16.0, abs=1.0)
    peaked = dsp.normalize_peak(x, target_db=-3.0)
    assert dsp.peak_db(peaked) == pytest.approx(-3.0, abs=0.1)


def test_soft_clip_bounds_the_signal():
    out = dsp.soft_clip(tone(440, 0.2, amp=4.0), drive=2.0)
    assert np.abs(out).max() <= 1.01


def test_denoise_lowers_a_broadband_floor():
    rng = np.random.default_rng(3)
    noisy = tone(440, 1.0, amp=0.3) + rng.standard_normal(SR).astype(np.float32) * 0.05
    out = dsp.denoise(noisy, SR, amount=0.6)
    assert dsp.rms_db(out) <= dsp.rms_db(noisy) + 0.5
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# vocal chain
# --------------------------------------------------------------------------
def test_vocal_master_is_produced_not_raw():
    raw = tone(220, 2.0, amp=0.05)
    out = mastering.master_vocal_only(raw, SR, VocalDirection(style="sad"))
    assert out.ndim == 2 and out.shape[1] == 2
    assert dsp.peak_db(out) <= -0.7                      # limiter ceiling
    assert dsp.loudness_db(out, SR) > dsp.loudness_db(dsp.as_stereo(raw), SR)
    assert np.isfinite(out).all()


def test_preview_is_lighter_than_the_master():
    direction = VocalDirection(style="romantic")
    preview = mastering.settings_for(direction, "preview")
    master = mastering.settings_for(direction, "master")
    assert preview.reverb_wet < master.reverb_wet
    assert preview.denoise <= master.denoise
    assert preview.target_loudness < master.target_loudness


def test_style_changes_the_chain():
    soft = mastering.settings_for(VocalDirection(style="soft"), "master")
    strong = mastering.settings_for(VocalDirection(style="strong"), "master")
    assert soft.reverb_wet > strong.reverb_wet
    assert strong.presence_gain > soft.presence_gain


def test_report_mentions_the_meters():
    text = mastering.report(dsp.as_stereo(tone(440, 0.5)), SR)
    assert "peak" in text and "loudness" in text
