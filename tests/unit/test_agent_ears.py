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


# --------------------------------------------------------------------------
# naming a pitch must not assume the answer
# --------------------------------------------------------------------------
def test_naming_constrained_to_a_raaga_cannot_hear_a_foreign_note(raagas):
    """Play a note the raaga does not contain, and it is renamed silently.

    This is the behaviour the ingestion path must not rely on: with the
    raaga passed, nearest_swara can only answer from that raaga's swaras.
    """
    hamsadhwani = raagas.require("Hamsadhwani")   # S R2 G3 P N3, no M1
    keeravani = raagas.require("Keeravani")       # has M1
    audio = render(keeravani, ["M1", "M1", "M1"], TONIC)

    snapped = analysis.analyse(audio, SR, hamsadhwani,
                               fixed_tonic_midi=float(TONIC))
    heard = {n.swara.rstrip("+-") for n in snapped.notes}
    assert "M1" not in heard, "the constrained naming should not emit M1"
    assert heard <= {"S", "R2", "G3", "P", "N3"}


def test_free_naming_hears_the_foreign_note_for_what_it_is(raagas):
    """The same audio, analysed free, keeps M1 - so a guard can reject it."""
    hamsadhwani = raagas.require("Hamsadhwani")
    keeravani = raagas.require("Keeravani")
    audio = render(keeravani, ["M1", "M1", "M1"], TONIC)

    free = analysis.analyse(audio, SR, hamsadhwani, fixed_tonic_midi=float(TONIC),
                            constrain_to_raaga=False)
    heard = {n.swara.rstrip("+-") for n in free.notes}
    assert "M1" in heard, "free naming must be able to say 'not in this raaga'"


def test_a_bent_note_keeps_the_raagas_name_but_a_foreign_one_does_not(raagas):
    """The middle ground the ingestion path relies on.

    Free naming is honest about foreign notes and brutal about ornament: a
    gamaka swings past a quarter tone, so a bent G3 gets called M1 and its
    phrase is discarded.  Re-labelling within a gamaka's reach keeps the
    ornament and still refuses a note that is really outside.
    """
    hamsadhwani = raagas.require("Hamsadhwani")   # S R2 G3 P N3
    tonic = float(TONIC)

    result = analysis.AnalysisResult(tonic_midi=tonic)
    result.notes = [
        # G3 is 4 semitones up; 60 cents sharp of it is a bent G3 ...
        analysis.AnalysedNote(start=0.0, duration=0.4, midi=tonic + 4.6),
        # ... while M1, a full semitone above G3, is not in this raaga.
        analysis.AnalysedNote(start=0.5, duration=0.4, midi=tonic + 5.0),
    ]
    for note in result.notes:
        note.swara, note.cents_deviation = analysis.nearest_swara(
            note.midi, tonic)                       # free naming, all twelve
    assert [n.swara for n in result.notes] == ["M1", "M1"]

    analysis.relabel_within_raaga(result, hamsadhwani)
    assert result.notes[0].swara == "G3", "a bent G3 should stay a G3"
    assert result.notes[1].swara == "M1", "a real M1 must stay foreign"


def test_a_flip_flopping_tracker_is_folded_back_into_the_line(raagas):
    """Octave doubling is the tracker's signature failure on a dense mix.

    "S- S++ S- S++" is autocorrelation locking onto a harmonic for a frame,
    not a singer leaping three octaves and back.  The swara is right and
    the register is wrong, so the note is moved, not invented.
    """
    tonic = float(TONIC)
    result = analysis.AnalysisResult(tonic_midi=tonic)
    notes = [
        analysis.AnalysedNote(start=0.0, duration=0.3, midi=tonic, swara="S"),
        analysis.AnalysedNote(start=0.3, duration=0.3, midi=tonic + 24,
                              swara="S++"),          # the tracker's octave slip
        analysis.AnalysedNote(start=0.6, duration=0.3, midi=tonic + 2,
                              swara="R2"),
        analysis.AnalysedNote(start=0.9, duration=0.3, midi=tonic + 4,
                              swara="G3"),
    ]
    result.notes = list(notes)
    result.phrases = [analysis.AnalysedPhrase(notes=result.notes, start=0.0,
                                              end=1.2)]
    assert analysis.widest_leap(result.phrases[0]) == 24

    moved = analysis.repair_octaves(result)
    assert moved == 1
    assert [n.swara for n in result.notes] == ["S", "S", "R2", "G3"], \
        "the swara must survive; only the octave was wrong"
    assert analysis.widest_leap(result.phrases[0]) <= 12


def test_a_line_that_still_leaps_is_left_leaping_to_be_refused(raagas):
    """Not every bad phrase is an octave problem, and this must not hide it."""
    tonic = float(TONIC)
    result = analysis.AnalysisResult(tonic_midi=tonic)
    result.notes = [
        analysis.AnalysedNote(start=0.0, duration=0.3, midi=tonic, swara="S"),
        analysis.AnalysedNote(start=0.3, duration=0.3, midi=tonic + 19,
                              swara="P+"),
    ]
    result.phrases = [analysis.AnalysedPhrase(notes=result.notes, start=0.0,
                                              end=0.6)]
    analysis.repair_octaves(result)
    # 19 semitones is not a whole number of octaves away from the median, so
    # moving octaves cannot fix it - and it stays visible to the guard.
    assert analysis.widest_leap(result.phrases[0]) > analysis.OCTAVE_LEAP_SEMITONES


def test_an_honest_octave_leap_is_left_alone(raagas):
    """S to S+ happens.  Only what sits outside the phrase gets moved."""
    tonic = float(TONIC)
    result = analysis.AnalysisResult(tonic_midi=tonic)
    result.notes = [
        analysis.AnalysedNote(start=0.0, duration=0.3, midi=tonic, swara="S"),
        analysis.AnalysedNote(start=0.3, duration=0.3, midi=tonic + 12,
                              swara="S+"),
    ]
    result.phrases = [analysis.AnalysedPhrase(notes=result.notes, start=0.0,
                                              end=0.6)]
    assert analysis.repair_octaves(result) == 0
    assert [n.swara for n in result.notes] == ["S", "S+"]


def test_the_tonic_help_survives_free_naming(raagas):
    """Only the naming goes free; the raaga still helps locate Sa."""
    raaga = raagas.require("Keeravani")
    audio = render(raaga, ["S", "R2", "G2", "M1", "P", "M1", "G2", "S"], 60)
    free = analysis.analyse(audio, SR, raaga, constrain_to_raaga=False)
    assert free.tonic_midi == pytest.approx(60, abs=1.0)


def test_resampling_preserves_the_pitch():
    t = np.arange(SR) / SR
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    resampled = analysis.resample(audio, SR, 16000)
    assert len(resampled) == 16000
    _, f0, _ = analysis.pitch_track(resampled, 16000)
    assert float(np.median(f0[f0 > 0])) == pytest.approx(220.0, rel=0.03)
