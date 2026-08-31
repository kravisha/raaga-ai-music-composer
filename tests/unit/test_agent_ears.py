"""Unit tests: the analysis pipeline - can the agent hear?"""
from __future__ import annotations

import numpy as np
import pytest

from raagacomposer.agent import analysis
from raagacomposer.core.models import Note
from raagacomposer.music import instruments as catalog
from raagacomposer.music.synth import render_notes
from raagacomposer.music.theory import midi_to_freq

pytestmark = pytest.mark.unit

SR = analysis.DEFAULT_SR
TONIC = 60


def render(raaga, swaras, tonic=TONIC, instrument="flute", duration=0.42,
           gap=0.12):
    inst = catalog.get(instrument)
    notes = []
    t = 0.0
    for swara in swaras:
        notes.append(Note(swara=swara, midi=raaga.midi(swara, tonic), start=t,
                          duration=duration, velocity=95))
        t += duration + gap
    return render_notes(notes, inst, SR, total_seconds=t + 0.3)


# --------------------------------------------------------------------------
# pitch
# --------------------------------------------------------------------------
def test_a_steady_tone_is_tracked():
    t = np.arange(int(SR * 1.0)) / SR
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    times, f0, conf = analysis.pitch_track(audio, SR)
    voiced = f0[f0 > 0]
    assert len(voiced) > 20
    assert float(np.median(voiced)) == pytest.approx(440.0, rel=0.02)
    assert float(np.mean(conf[f0 > 0])) > 0.5


def test_silence_and_noise_produce_no_confident_pitch():
    assert not np.any(analysis.pitch_track(np.zeros(SR, np.float32), SR)[1] > 0)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.3
    _, f0, conf = analysis.pitch_track(noise, SR)
    assert float(np.mean(f0 > 0)) < 0.6


def test_voiced_spans_find_the_notes():
    t = np.arange(int(SR * 0.5)) / SR
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    audio = np.concatenate([np.zeros(int(SR * 0.3), np.float32), tone,
                            np.zeros(int(SR * 0.3), np.float32), tone])
    spans = analysis.voiced_spans(audio, SR)
    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(0.3, abs=0.1)


# --------------------------------------------------------------------------
# tonic
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tonic", [55, 60, 65])
def test_the_tonic_is_found_from_a_phrase(raagas, tonic):
    raaga = raagas.require("Keeravani")
    audio = render(raaga, ["S", "R2", "G2", "M1", "P", "M1", "G2", "S"], tonic)
    result = analysis.analyse(audio, SR, raaga)
    assert result.tonic_midi == pytest.approx(tonic, abs=1.0)
    assert result.confidence > 0.4


def test_a_given_tonic_is_used_as_given(raagas):
    raaga = raagas.require("Keeravani")
    audio = render(raaga, ["P"], TONIC, duration=0.8)
    result = analysis.analyse(audio, SR, raaga, fixed_tonic_midi=float(TONIC))
    assert result.tonic_midi == pytest.approx(TONIC)
    assert result.notes[0].swara == "P"


def test_naming_one_note_needs_the_drone(raagas):
    """Without a given Sa a single note is heard as the tonic itself."""
    raaga = raagas.require("Keeravani")
    audio = render(raaga, ["P"], TONIC, duration=0.8)
    guessed = analysis.analyse(audio, SR, raaga)
    given = analysis.analyse(audio, SR, raaga, fixed_tonic_midi=float(TONIC))
    assert given.notes[0].swara == "P"
    assert guessed.notes[0].swara != given.notes[0].swara


# --------------------------------------------------------------------------
# swara mapping
# --------------------------------------------------------------------------
def test_pitches_map_to_the_right_swara(raagas):
    raaga = raagas.require("Keeravani")
    for swara in raaga.allowed:
        midi = raaga.midi(swara, TONIC)
        heard, cents = analysis.nearest_swara(midi, TONIC, raaga)
        assert heard == swara
        assert abs(cents) < 1.0


def test_the_octave_above_is_named_as_such(raagas):
    raaga = raagas.require("Keeravani")
    heard, _ = analysis.nearest_swara(TONIC + 12, TONIC, raaga)
    assert heard == "S+"
    heard, _ = analysis.nearest_swara(TONIC - 12, TONIC, raaga)
    assert heard == "S-"


def test_a_note_just_below_the_upper_sa_rounds_up(raagas):
    """A slightly flat upper Sa must not be heard as the leading note."""
    raaga = raagas.require("Keeravani")
    heard, _ = analysis.nearest_swara(TONIC + 11.6, TONIC, raaga)
    assert heard == "S+"


@pytest.mark.parametrize("name,instrument,swaras", [
    ("Keeravani", "flute", ["S", "R2", "G2", "M1", "P", "D1", "N3", "S+"]),
    ("Keeravani", "veena", ["S+", "N3", "D1", "P", "M1", "G2", "R2", "S"]),
    ("Mohanam", "violin", ["S", "R2", "G3", "P", "D2", "S+"]),
    ("Kalyani", "flute", ["G3", "M2", "P", "D2", "N3", "S+"]),
])
def test_a_played_phrase_is_transcribed(raagas, name, instrument, swaras):
    raaga = raagas.require(name)
    result = analysis.analyse(render(raaga, swaras, instrument=instrument), SR,
                              raaga, fixed_tonic_midi=float(TONIC))
    assert result.swara_sequence() == swaras
    assert result.confidence > 0.5


def test_note_timings_are_recovered(raagas):
    raaga = raagas.require("Keeravani")
    result = analysis.analyse(render(raaga, ["S", "R2", "G2"]), SR, raaga,
                              fixed_tonic_midi=float(TONIC))
    assert len(result.notes) == 3
    assert result.notes[0].start == pytest.approx(0.0, abs=0.08)
    assert result.notes[1].start == pytest.approx(0.54, abs=0.12)
    assert all(n.duration > 0.2 for n in result.notes)


# --------------------------------------------------------------------------
# phrases and tempo
# --------------------------------------------------------------------------
def test_phrases_are_split_at_the_breaths(raagas):
    raaga = raagas.require("Keeravani")
    first = render(raaga, ["S", "R2", "G2"])
    silence = np.zeros(int(SR * 0.6), dtype=np.float32)
    second = render(raaga, ["P", "D1", "N3"])
    audio = np.concatenate([first, silence, second])
    result = analysis.analyse(audio, SR, raaga, fixed_tonic_midi=float(TONIC))
    assert len(result.phrases) >= 2
    assert result.phrases[0].swaras[:3] == ["S", "R2", "G2"]


def test_phrase_shape_and_contour(raagas):
    raaga = raagas.require("Keeravani")
    rising = analysis.analyse(render(raaga, ["S", "R2", "G2", "M1"]), SR, raaga,
                              fixed_tonic_midi=float(TONIC)).phrases[0]
    falling = analysis.analyse(render(raaga, ["M1", "G2", "R2", "S"]), SR, raaga,
                               fixed_tonic_midi=float(TONIC)).phrases[0]
    assert rising.shape() == "rise"
    assert falling.shape() == "fall"
    assert set(rising.contour()) <= {"u", "d", "r"}


def test_tempo_is_estimated(raagas):
    raaga = raagas.require("Keeravani")
    bpm = 96
    beat = 60.0 / bpm
    notes = [Note(swara="S", midi=raaga.midi(s, TONIC), start=i * beat,
                  duration=beat * 0.6)
             for i, s in enumerate(["S", "R2", "G2", "M1", "P", "M1", "G2", "S"])]
    audio = render_notes(notes, catalog.get("piano"), SR,
                         total_seconds=len(notes) * beat + 0.5)
    result = analysis.analyse(audio, SR, raaga, fixed_tonic_midi=float(TONIC))
    candidates = [result.tempo_bpm, result.tempo_bpm * 2, result.tempo_bpm / 2]
    assert min(abs(c - bpm) for c in candidates) < bpm * 0.2


def test_phrase_features_describe_the_shape(raagas):
    raaga = raagas.require("Keeravani")
    phrase = analysis.analyse(render(raaga, ["S", "R2", "G2", "M1"]), SR, raaga,
                              fixed_tonic_midi=float(TONIC)).phrases[0]
    features = analysis.phrase_features(phrase)
    assert features["length"] == 4
    assert features["starts_on"] == "S"
    assert features["ends_on"] == "M1"
    assert features["span"] > 0


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------
def test_silence_is_reported_not_invented():
    result = analysis.analyse(np.zeros(SR, np.float32), SR)
    assert result.notes == []
    assert result.confidence == 0.0
    assert any("silent" in w for w in result.warnings)


def test_audio_that_is_too_short_is_reported():
    result = analysis.analyse(np.zeros(100, np.float32), SR)
    assert any("short" in w for w in result.warnings)


def test_noise_yields_no_confident_transcription():
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(int(SR * 1.5)).astype(np.float32) * 0.3
    result = analysis.analyse(noise, SR)
    assert result.confidence < 0.6


def test_a_file_can_be_analysed(tmp_path, raagas):
    import soundfile as sf
    raaga = raagas.require("Keeravani")
    path = tmp_path / "phrase.wav"
    sf.write(str(path), render(raaga, ["S", "R2", "G2", "M1"]), SR)
    result = analysis.analyse_file(path, raaga, tonic_hint_midi=float(TONIC))
    assert len(result.notes) == 4
    assert result.version == analysis.ANALYSIS_VERSION


def test_resampling_preserves_the_pitch():
    t = np.arange(SR) / SR
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    resampled = analysis.resample(audio, SR, 16000)
    assert len(resampled) == 16000
    _, f0, _ = analysis.pitch_track(resampled, 16000)
    assert float(np.median(f0[f0 > 0])) == pytest.approx(220.0, rel=0.03)
