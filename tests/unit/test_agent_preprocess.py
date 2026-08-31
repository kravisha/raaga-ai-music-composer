"""Preparing a real recording for the ears.

Two jobs, tested separately: finding the drone and taking it out, and telling
singing from speech.  Every signal here is synthesised, so what the answer
should be is known exactly rather than judged by ear.
"""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.agent import preprocess as P
from tests.conftest import (ANALYSIS_SR, drone_signal, gamaka_signal,
                            lesson_signal, speech_signal, sung_signal)

pytestmark = pytest.mark.unit


def cents(a: float, b: float) -> float:
    return 1200.0 * np.log2(a / b)


# --------------------------------------------------------------------------
# finding the drone
# --------------------------------------------------------------------------
def test_a_tanpura_is_found():
    drone = P.detect_drone(drone_signal(6.0), ANALYSIS_SR)
    assert drone.found
    assert abs(cents(drone.hz, 261.63)) < 25
    assert drone.confidence > 0.5


def test_singing_alone_is_not_mistaken_for_a_drone():
    """Nothing sustained is running under it, so there is nothing to notch."""
    drone = P.detect_drone(sung_signal(6.0), ANALYSIS_SR)
    assert not drone.found


def test_the_drone_is_found_underneath_singing():
    mixed = 0.7 * sung_signal(6.0) + 0.6 * drone_signal(6.0)
    drone = P.detect_drone(mixed.astype(np.float32), ANALYSIS_SR)
    assert drone.found
    assert abs(cents(drone.hz, 261.63)) < 30


@pytest.mark.parametrize("sa_hz", [130.81, 164.81, 196.0, 220.0, 261.63])
def test_the_tonic_is_accurate_enough_to_measure_swaras_from(sa_hz: float):
    """A bin is most of a semitone wide at these pitches.  Every swara is
    measured from this number, so it has to be far better than a bin."""
    mixed = 0.7 * sung_signal(6.0) + 0.6 * drone_signal(6.0, sa_hz=sa_hz)
    drone = P.detect_drone(mixed.astype(np.float32), ANALYSIS_SR)
    assert drone.found
    assert abs(cents(drone.hz, sa_hz)) < 25


def test_notching_removes_the_drone_and_leaves_the_singing():
    voice, drone_only = sung_signal(6.0), drone_signal(6.0)
    mixed = (0.7 * voice + 0.6 * drone_only).astype(np.float32)
    estimate = P.detect_drone(mixed, ANALYSIS_SR)
    cleaned = P.suppress_drone(mixed, ANALYSIS_SR, estimate)

    def energy_at(signal: np.ndarray, freq: float) -> float:
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        freqs = np.fft.rfftfreq(len(signal), 1.0 / ANALYSIS_SR)
        band = (freqs > freq - 6) & (freqs < freq + 6)
        return float(spectrum[band].sum())

    before, after = energy_at(mixed, 261.63), energy_at(cleaned, 261.63)
    assert after < before * 0.5, "the drone's Sa should be much quieter"
    assert float(np.abs(cleaned).max()) > 0.1, "the recording is still there"


def test_no_drone_means_the_audio_is_returned_untouched():
    voice = sung_signal(4.0)
    same = P.suppress_drone(voice, ANALYSIS_SR, P.DroneEstimate())
    assert same is voice


# --------------------------------------------------------------------------
# telling singing from speech
# --------------------------------------------------------------------------
def test_singing_scores_as_singing():
    spans = P.sung_spans(sung_signal(6.0), ANALYSIS_SR)
    assert spans and all(s.is_sung for s in spans)


def test_speech_scores_as_speech():
    spans = P.sung_spans(speech_signal(6.0), ANALYSIS_SR)
    assert spans and not any(s.is_sung for s in spans)


def test_the_two_are_separated_by_a_clear_margin():
    """A gate that only just separates them would not survive a real room."""
    sung = np.mean([s.music_likelihood
                    for s in P.sung_spans(sung_signal(6.0), ANALYSIS_SR)])
    spoken = np.mean([s.music_likelihood
                      for s in P.sung_spans(speech_signal(6.0), ANALYSIS_SR)])
    assert sung - spoken > 0.3, f"margin was only {sung - spoken:.2f}"


def _plateau_of(signal) -> float:
    _, _, plateau = P.score_window(
        P.analysis.pitch_track(signal, ANALYSIS_SR)[1], P.analysis.HOP_SECONDS)
    return plateau


def test_held_pitch_is_what_separates_them():
    """The feature the gate rests on, measured on its own.

    Speech is not perfectly unmusical - at a turning point in its contour it
    does level off - so what is asserted is the margin, not a small absolute
    number that would be luck rather than evidence.
    """
    sung_plateau = _plateau_of(sung_signal(4.0))
    spoken_plateau = _plateau_of(speech_signal(4.0))
    assert sung_plateau > 0.85
    assert sung_plateau - spoken_plateau > 0.5


def test_heavy_gamaka_still_reads_as_singing():
    """The failure this feature must not have.  A kampita swinging most of a
    semitone leaves the raw contour anything but flat; if the gate judged
    flatness directly it would throw away exactly the ornamented singing that
    is worth learning from."""
    for swing in (60.0, 90.0, 140.0):
        spans = P.sung_spans(gamaka_signal(8.0, swing_cents=swing), ANALYSIS_SR)
        assert spans and all(s.is_sung for s in spans), \
            f"gamaka of +-{swing:.0f} cents was gated out as speech"


def test_an_isolated_moment_of_steady_pitch_is_not_a_phrase():
    """Speech levelling off for a second is not singing.  A run has to last
    long enough to be a phrase before any of it is kept."""
    spans = P.sung_spans(speech_signal(8.0), ANALYSIS_SR)
    assert spans and not any(s.is_sung for s in spans)


# --------------------------------------------------------------------------
# the whole preparation
# --------------------------------------------------------------------------
def test_a_lesson_keeps_the_singing_and_silences_the_talking():
    audio, sung_start, sung_end = lesson_signal(talk_seconds=4.0,
                                                sung_seconds=6.0)
    prepared = P.prepare(audio, ANALYSIS_SR)

    assert prepared.drone.found
    assert prepared.silenced_seconds > 3.0, "the talking should be silenced"
    assert prepared.kept_seconds > 5.0, "the singing should survive"

    def rms(start: float, end: float) -> float:
        block = prepared.audio[int(start * ANALYSIS_SR):int(end * ANALYSIS_SR)]
        return float(np.sqrt(np.mean(block ** 2))) if len(block) else 0.0

    assert rms(sung_start + 1.0, sung_end - 1.0) > 10 * rms(0.5, 2.5), \
        "what is left should be the sung stretch, not the spoken one"


def test_preparation_never_splices_the_recording():
    """Rejected stretches are silenced in place.  Cutting them out would move
    every timestamp after the cut and break phrase segmentation."""
    audio, _, _ = lesson_signal()
    prepared = P.prepare(audio, ANALYSIS_SR)
    assert len(prepared.audio) == len(audio)


def test_the_drone_supplies_the_tonic():
    audio, _, _ = lesson_signal(sa_hz=196.0)
    prepared = P.prepare(audio, ANALYSIS_SR)
    assert prepared.tonic_midi is not None
    assert abs(prepared.tonic_midi - 55.0) < 0.35     # G3 = MIDI 55


def test_an_all_speech_recording_is_left_alone_rather_than_emptied():
    """If the gate rejects everything, that is far more likely to be the gate
    failing to suit the recording than a lesson with no singing in it.  It
    stands down and says so instead of handing back silence."""
    audio = (0.75 * speech_signal(8.0)
             + 0.55 * drone_signal(8.0)).astype(np.float32)
    prepared = P.prepare(audio, ANALYSIS_SR)
    assert prepared.silenced_seconds == 0.0
    assert any("whole recording" in w for w in prepared.warnings)


def test_the_gate_can_be_turned_off_independently():
    audio, _, _ = lesson_signal()
    prepared = P.prepare(audio, ANALYSIS_SR, gate_speech=False)
    assert prepared.silenced_seconds == 0.0
    assert prepared.drone.found, "the drone is still handled"


def test_silence_and_very_short_audio_are_survivable():
    quiet = P.prepare(np.zeros(ANALYSIS_SR * 2, dtype=np.float32), ANALYSIS_SR)
    assert "silent" in " ".join(quiet.warnings)

    tiny = P.prepare(np.zeros(100, dtype=np.float32), ANALYSIS_SR)
    assert "short" in " ".join(tiny.warnings)
