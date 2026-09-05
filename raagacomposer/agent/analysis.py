"""Audio ingestion and music analysis - the agent's ears.

Learning specification section 7.  The pipeline is:

    audio -> mono, resample -> silence detection -> pitch contour ->
    tonic estimation -> swara mapping -> phrase segmentation -> features

Nothing here needs a perfect transcription.  It needs to be honest about how
sure it is, because confidence is what the knowledge repository stores and what
the curriculum thresholds act on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging_setup import get_logger
from ..music.theory import freq_to_midi, midi_to_freq
from ..raaga.library import Raaga, SWARA_SEMITONES

log = get_logger("agent.analysis")

#: What the ears are, as a version.  Every source row records the version
#: that derived it, so knowledge made by code that no longer exists can be
#: found and rebuilt rather than trusted.
#:
#: **Bump this whenever the extraction changes what it produces.**  Not for
#: a refactor, a rename or a speed-up - for anything that would make the
#: same recording yield different phrases.  Leaving it still is how a
#: knowledge base quietly becomes a mixture of several codebases.
#:
#: 1 -> 2: swara naming may range over all twelve for material from outside
#: (``constrain_to_raaga``), notes within a gamaka's reach are given the
#: raaga's own name (``relabel_within_raaga``), octave slips are folded back
#: into the phrase (``repair_octaves``), and a phrase still leaping more
#: than an octave is refused.
ANALYSIS_VERSION = "pitch-autocorr-2"

DEFAULT_SR = 22050
FRAME_SECONDS = 0.046
HOP_SECONDS = 0.012
F_MIN = 70.0
F_MAX = 1100.0


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class AnalysedNote:
    start: float
    duration: float
    midi: float
    swara: str = ""
    confidence: float = 0.0
    cents_deviation: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class AnalysedPhrase:
    notes: List[AnalysedNote] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0

    @property
    def swaras(self) -> List[str]:
        return [n.swara for n in self.notes if n.swara]

    @property
    def midi(self) -> List[int]:
        return [int(round(n.midi)) for n in self.notes]

    @property
    def durations(self) -> List[float]:
        return [round(n.duration, 3) for n in self.notes]

    def contour(self) -> str:
        """Coarse shape: u(p), d(own), r(epeat), one letter per step."""
        out = []
        for a, b in zip(self.notes, self.notes[1:]):
            delta = b.midi - a.midi
            out.append("u" if delta > 0.5 else ("d" if delta < -0.5 else "r"))
        return "".join(out)

    def shape(self) -> str:
        contour = self.contour()
        if not contour:
            return "flat"
        ups, downs = contour.count("u"), contour.count("d")
        if ups and not downs:
            return "rise"
        if downs and not ups:
            return "fall"
        first_half = contour[: max(1, len(contour) // 2)]
        return "arch" if first_half.count("u") >= first_half.count("d") else "valley"


@dataclass
class AnalysisResult:
    tonic_hz: float = 0.0
    tonic_midi: float = 0.0
    tempo_bpm: float = 0.0
    duration: float = 0.0
    sample_rate: int = DEFAULT_SR
    notes: List[AnalysedNote] = field(default_factory=list)
    phrases: List[AnalysedPhrase] = field(default_factory=list)
    voiced_ratio: float = 0.0
    confidence: float = 0.0
    version: str = ANALYSIS_VERSION
    warnings: List[str] = field(default_factory=list)

    def swara_sequence(self) -> List[str]:
        return [n.swara for n in self.notes if n.swara]

    def summary(self) -> str:
        return (f"tonic {self.tonic_hz:.1f} Hz (MIDI {self.tonic_midi:.1f}), "
                f"{len(self.notes)} notes, {len(self.phrases)} phrases, "
                f"{self.tempo_bpm:.0f} bpm, confidence {self.confidence:.2f}")


# --------------------------------------------------------------------------
# loading and normalisation
# --------------------------------------------------------------------------
def load_audio(path: Path, target_sr: int = DEFAULT_SR) -> Tuple[np.ndarray, int]:
    """Decode a file to mono float32 at the analysis sample rate."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return resample(mono, sr, target_sr), target_sr


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src == dst or len(audio) == 0:
        return audio
    n = int(round(len(audio) * dst / src))
    return np.interp(np.linspace(0, len(audio) - 1, n),
                     np.arange(len(audio)), audio).astype(np.float32)


def voiced_spans(audio: np.ndarray, sr: int, threshold_db: float = -42.0,
                 min_seconds: float = 0.06) -> List[Tuple[float, float]]:
    """Energy gate: the spans where something is actually sounding."""
    frame = max(1, int(0.02 * sr))
    hop = max(1, frame // 2)
    if len(audio) < frame:
        return []
    frames = 1 + (len(audio) - frame) // hop
    rms = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        block = audio[i * hop:i * hop + frame]
        rms[i] = float(np.sqrt(np.mean(block ** 2)))
    peak = float(rms.max()) if frames else 0.0
    if peak <= 1e-6:
        return []
    db = 20 * np.log10(np.maximum(rms / peak, 1e-6))
    loud = db > threshold_db

    spans: List[Tuple[float, float]] = []
    start = None
    for i, on in enumerate(loud):
        t = i * hop / sr
        if on and start is None:
            start = t
        elif not on and start is not None:
            if t - start >= min_seconds:
                spans.append((start, t))
            start = None
    if start is not None:
        end = len(audio) / sr
        if end - start >= min_seconds:
            spans.append((start, end))
    return spans


# --------------------------------------------------------------------------
# pitch tracking
# --------------------------------------------------------------------------
def pitch_track(audio: np.ndarray, sr: int, hop_seconds: float = HOP_SECONDS,
                frame_seconds: float = FRAME_SECONDS,
                f_min: float = F_MIN, f_max: float = F_MAX
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Autocorrelation pitch tracker.

    Returns (times, f0 in Hz with 0 for unvoiced, per-frame confidence 0..1).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = int(frame_seconds * sr)
    hop = max(1, int(hop_seconds * sr))
    if len(audio) < frame or frame < 32:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    lag_min = max(2, int(sr / f_max))
    lag_max = min(frame - 2, int(sr / f_min))
    if lag_max <= lag_min:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    count = 1 + (len(audio) - frame) // hop
    times = np.arange(count, dtype=np.float32) * hop / sr
    f0 = np.zeros(count, dtype=np.float32)
    conf = np.zeros(count, dtype=np.float32)
    window = np.hanning(frame).astype(np.float32)

    for i in range(count):
        block = audio[i * hop:i * hop + frame] * window
        energy = float(np.sqrt(np.mean(block ** 2)))
        if energy < 1e-4:
            continue
        block = block - block.mean()
        corr = np.correlate(block, block, mode="full")[frame - 1:]
        zero = corr[0]
        if zero <= 1e-9:
            continue
        segment = corr[lag_min:lag_max]
        if len(segment) < 3:
            continue
        peak = int(np.argmax(segment)) + lag_min
        strength = float(corr[peak] / zero)
        if strength < 0.3:
            continue
        # Parabolic interpolation around the peak for sub-sample accuracy.
        if 1 <= peak < len(corr) - 1:
            a, b, c = corr[peak - 1], corr[peak], corr[peak + 1]
            denom = (a - 2 * b + c)
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            peak_f = peak + float(np.clip(shift, -1.0, 1.0))
        else:
            peak_f = float(peak)
        freq = sr / max(1e-6, peak_f)
        if f_min <= freq <= f_max:
            f0[i] = freq
            conf[i] = min(1.0, strength)
    return times, f0, conf


# --------------------------------------------------------------------------
# tonic and swara mapping
# --------------------------------------------------------------------------
def estimate_tonic(f0: np.ndarray, weights: Optional[np.ndarray] = None,
                   raaga: Optional[Raaga] = None,
                   hint_midi: Optional[float] = None) -> Tuple[float, float]:
    """Estimate the tonic. Returns (hz, confidence).

    Pitch mass is folded into one octave; the tonic is the pitch class that best
    explains the rest of the material.  When a raaga is supplied, the candidate
    that puts the most mass on that raaga's own intervals wins - which is how a
    listener finds Sa: by hearing which note makes the others make sense.
    """
    voiced = f0 > 0
    if not np.any(voiced):
        return 0.0, 0.0
    midi = np.array([freq_to_midi(float(f)) for f in f0[voiced]], dtype=np.float32)
    mass = (weights[voiced] if weights is not None
            else np.ones(len(midi), dtype=np.float32))

    # 12-bin pitch-class histogram at quarter-tone resolution.
    bins = np.zeros(48, dtype=np.float64)
    for value, w in zip(midi, mass):
        idx = int(round((value % 12.0) * 4)) % 48
        bins[idx] += float(w)
    if bins.sum() <= 0:
        return 0.0, 0.0
    bins /= bins.sum()

    intervals = [0, 2, 4, 5, 7, 9, 11]
    if raaga is not None:
        intervals = sorted({SWARA_SEMITONES.get(s, 0) for s in raaga.allowed})

    best_class, best_score = 0.0, -1.0
    for candidate in range(48):
        score = 0.0
        for semitone in intervals:
            target = (candidate + semitone * 4) % 48
            score += bins[target] + 0.5 * bins[(target - 1) % 48] \
                + 0.5 * bins[(target + 1) % 48]
        score += 0.6 * bins[candidate]          # the tonic itself should be sung
        if score > best_score:
            best_score, best_class = score, candidate / 4.0

    median = float(np.median(midi))
    tonic_midi = best_class + 12.0 * math.floor((median - best_class) / 12.0)
    while tonic_midi > median:
        tonic_midi -= 12.0
    if hint_midi is not None:
        # Choose the octave nearest a supplied expectation without moving the
        # pitch class the analysis actually found.
        while tonic_midi + 12 <= hint_midi + 6:
            tonic_midi += 12
        while tonic_midi - 12 >= hint_midi - 6:
            tonic_midi -= 12

    confidence = float(min(1.0, max(0.0, (best_score - 1.0 / len(intervals)) * 1.2)))
    return midi_to_freq(tonic_midi), confidence


def nearest_swara(midi_value: float, tonic_midi: float,
                  raaga: Optional[Raaga] = None) -> Tuple[str, float]:
    """Map a pitch to a swara name. Returns (token, cents deviation)."""
    semitones = midi_value - tonic_midi
    allowed = raaga.allowed if raaga is not None else \
        ["S", "R2", "G3", "M1", "P", "D2", "N3"]
    # Search absolute targets across octaves: a note a little flat of the upper
    # Sa must round to S+, not to the nearest swara inside the lower octave.
    best_swara, best_octave, best_delta = "S", 0, 1e9
    for octave in range(-4, 5):
        for swara in allowed:
            target = SWARA_SEMITONES.get(swara, 0) + 12 * octave
            delta = semitones - target
            if abs(delta) < abs(best_delta):
                best_swara, best_octave, best_delta = swara, octave, delta
    marks = "+" * best_octave if best_octave > 0 else "-" * (-best_octave)
    return best_swara + marks, best_delta * 100.0


def notes_from_pitch(times: np.ndarray, f0: np.ndarray, conf: np.ndarray,
                     tonic_midi: float, raaga: Optional[Raaga] = None,
                     min_duration: float = 0.07) -> List[AnalysedNote]:
    """Group a pitch contour into discrete notes."""
    notes: List[AnalysedNote] = []
    if len(times) == 0:
        return notes

    current_semitone: Optional[int] = None
    start = 0.0
    values: List[float] = []
    confs: List[float] = []

    def flush(end: float) -> None:
        nonlocal values, confs, current_semitone
        if current_semitone is not None and values:
            duration = end - start
            if duration >= min_duration:
                midi_value = float(np.median(values))
                swara, cents = nearest_swara(midi_value, tonic_midi, raaga)
                notes.append(AnalysedNote(
                    start=round(start, 4), duration=round(duration, 4),
                    midi=round(midi_value, 3), swara=swara,
                    confidence=float(np.mean(confs)) if confs else 0.0,
                    cents_deviation=round(cents, 1)))
        values, confs, current_semitone = [], [], None

    hop = float(times[1] - times[0]) if len(times) > 1 else HOP_SECONDS
    for i, t in enumerate(times):
        if f0[i] <= 0:
            flush(float(t))
            continue
        midi_value = freq_to_midi(float(f0[i]))
        semitone = int(round(midi_value - tonic_midi))
        if current_semitone is None:
            current_semitone = semitone
            start = float(t)
        elif semitone != current_semitone:
            flush(float(t))
            current_semitone = semitone
            start = float(t)
        values.append(midi_value)
        confs.append(float(conf[i]))
    flush(float(times[-1]) + hop)
    return notes


def segment_phrases(notes: Sequence[AnalysedNote], gap: float = 0.22,
                    max_notes: int = 12) -> List[AnalysedPhrase]:
    """Split notes into breath-separated phrases."""
    phrases: List[AnalysedPhrase] = []
    current: List[AnalysedNote] = []
    for i, note in enumerate(notes):
        current.append(note)
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        boundary = nxt is None or (nxt.start - note.end) >= gap \
            or len(current) >= max_notes
        if boundary and current:
            phrases.append(AnalysedPhrase(
                notes=list(current), start=current[0].start,
                end=current[-1].end,
                confidence=float(np.mean([n.confidence for n in current]))))
            current = []
    return phrases


def estimate_tempo(notes: Sequence[AnalysedNote]) -> float:
    """Tempo from the most common inter-onset interval."""
    if len(notes) < 4:
        return 0.0
    gaps = np.array([b.start - a.start for a, b in zip(notes, notes[1:])],
                    dtype=np.float32)
    gaps = gaps[(gaps > 0.08) & (gaps < 3.0)]
    if len(gaps) < 3:
        return 0.0
    # Cluster onto a coarse grid and take the strongest bucket.
    grid = np.round(gaps / 0.02) * 0.02
    values, counts = np.unique(grid, return_counts=True)
    beat = float(values[int(np.argmax(counts))])
    if beat <= 0:
        return 0.0
    bpm = 60.0 / beat
    while bpm < 40:
        bpm *= 2
    while bpm > 200:
        bpm /= 2
    return round(bpm, 1)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------
def analyse(audio: np.ndarray, sr: int, raaga: Optional[Raaga] = None,
            tonic_hint_midi: Optional[float] = None,
            fixed_tonic_midi: Optional[float] = None,
            constrain_to_raaga: bool = True) -> AnalysisResult:
    """Run the whole pipeline over an audio buffer.

    ``fixed_tonic_midi`` is the tanpura: when the tonic is given rather than
    inferred, a single note can be named, which estimation from one pitch
    could never do.

    ``raaga`` does two separate jobs here, and a caller checking whether a
    recording is *in* that raaga must not want both.  It helps locate Sa,
    which is a fair use of what you know.  It also restricts the names a
    pitch may be given, and that is question-begging: with a raaga passed,
    ``nearest_swara`` can only answer from that raaga's swaras, so every
    note lands inside it and conformance becomes 100% by arithmetic.  Pass
    ``constrain_to_raaga=False`` to keep the tonic help and let the naming
    range over all twelve, so that "is this really Hamsadhwani?" has an
    answer that could have been no.
    """
    result = AnalysisResult(sample_rate=sr)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    result.duration = len(audio) / max(1, sr)
    if result.duration < 0.2:
        result.warnings.append("audio too short to analyse")
        return result

    peak = float(np.abs(audio).max())
    if peak < 1e-4:
        result.warnings.append("audio is silent")
        return result
    audio = audio / peak

    times, f0, conf = pitch_track(audio, sr)
    if len(times) == 0:
        result.warnings.append("no frames to analyse")
        return result
    result.voiced_ratio = float(np.mean(f0 > 0))
    if result.voiced_ratio < 0.05:
        result.warnings.append("almost nothing pitched was found")
        return result

    if fixed_tonic_midi is not None:
        tonic_hz, tonic_conf = midi_to_freq(fixed_tonic_midi), 1.0
    else:
        tonic_hz, tonic_conf = estimate_tonic(f0, conf, raaga, tonic_hint_midi)
    if tonic_hz <= 0:
        result.warnings.append("could not find a tonic")
        return result
    result.tonic_hz = tonic_hz
    result.tonic_midi = (float(fixed_tonic_midi) if fixed_tonic_midi is not None
                         else freq_to_midi(tonic_hz))

    # The raaga has already helped find Sa above; from here it would only
    # be deciding the answer to the question being asked.
    result.notes = notes_from_pitch(times, f0, conf, result.tonic_midi,
                                    raaga if constrain_to_raaga else None)
    result.phrases = segment_phrases(result.notes)
    result.tempo_bpm = estimate_tempo(result.notes)

    note_conf = float(np.mean([n.confidence for n in result.notes])) \
        if result.notes else 0.0
    tuning = 1.0
    if result.notes:
        deviations = np.array([abs(n.cents_deviation) for n in result.notes])
        tuning = float(np.clip(1.0 - deviations.mean() / 50.0, 0.0, 1.0))
    result.confidence = round(
        0.4 * tonic_conf + 0.35 * note_conf + 0.25 * tuning, 3)
    return result


#: A melodic line does not leap this far between consecutive notes.  An
#: autocorrelation tracker, on the other hand, does it constantly on a dense
#: mix: it locks onto a harmonic for a frame or two and reports a note an
#: octave or two away from the one being sung.  Measured on a film-song
#: mashup, half the phrases carried a leap of an octave or more and a third
#: exceeded two - "S- S++ S- S++ G3 S++" is the tracker flip-flopping, not
#: a singer.
OCTAVE_LEAP_SEMITONES = 12.0


def _shift_octave(swara: str, steps: int) -> str:
    """Move a swara name by whole octaves, keeping which swara it is."""
    base = swara.rstrip("+-")
    marks = swara[len(base):]
    octave = marks.count("+") - marks.count("-") + steps
    return base + ("+" * octave if octave > 0 else "-" * (-octave))


def repair_octaves(result: AnalysisResult) -> int:
    """Fold notes the tracker put in the wrong octave back into the line.

    The pitch *class* is the part autocorrelation gets right - the swara is
    a real swara - and the octave is the part it gets wrong.  So this moves
    a stray note by whole octaves towards the middle of its own phrase,
    which changes the register without inventing a note that was not sung.

    A phrase that is still leaping after this cannot be repaired by moving
    octaves around, and ``research.py`` refuses it rather than teaching the
    composer a contour nobody played.

    Returns the number of notes moved.  Mutates in place; phrases and
    ``result.notes`` hold the same objects.
    """
    import statistics

    moved = 0
    for phrase in result.phrases:
        notes = phrase.notes
        if len(notes) < 2:
            continue
        middle = statistics.median(n.midi for n in notes)
        for note in notes:
            if abs(note.midi - middle) <= OCTAVE_LEAP_SEMITONES:
                continue
            # Move it to whichever octave sits closest to the middle of the
            # phrase, rather than only just inside an octave of it: a note
            # two octaves out is not repaired by bringing it one closer.
            steps = int(round((middle - note.midi) / 12.0))
            if not steps:
                continue
            note.midi += 12.0 * steps
            note.swara = _shift_octave(note.swara, steps)
            moved += 1
    return moved


def widest_leap(phrase: AnalysedPhrase) -> float:
    """The largest step between consecutive notes, in semitones."""
    notes = phrase.notes
    if len(notes) < 2:
        return 0.0
    return max(abs(b.midi - a.midi) for a, b in zip(notes, notes[1:]))


#: How far a note may sit from one of the raaga's swaras and still be that
#: swara.  Carnatic phrasing is not a sequence of held pitches: a gamaka
#: swings well past a quarter tone, so judging by the nearest of twelve
#: names throws away real music - a G2 bent 60 cents sharp gets called G3
#: and its whole phrase is discarded.  Two swaras are never closer than a
#: semitone, so 70 cents keeps bent notes and still refuses a foreign one.
GAMAKA_TOLERANCE_CENTS = 70.0


def relabel_within_raaga(result: AnalysisResult, raaga: Raaga,
                         tolerance_cents: float = GAMAKA_TOLERANCE_CENTS
                         ) -> AnalysisResult:
    """Let freely-named notes take the raaga's name when they are close.

    This is the middle ground between the two ways of naming a pitch, and
    it exists because both extremes are wrong for judging whether a
    recording is in a raaga.

    Naming *with* the raaga answers the question before it is asked: every
    pitch snaps to one of that raaga's swaras, conformance is 100% by
    arithmetic, and a guard downstream can never reject anything.  Naming
    freely across all twelve is honest about foreign notes but brutal about
    ornament, because a bent note lands on a neighbour's name.

    So: name freely first, then give a note the raaga's own name when its
    pitch is within ``tolerance_cents`` of one of the raaga's swaras.  What
    is left outside is genuinely outside - at least a semitone from every
    note the raaga has - and the guard can act on it.

    Mutates ``result`` in place; phrases hold the same note objects.
    """
    allowed = list(raaga.allowed)
    if not allowed:
        return result
    for note in result.notes:
        if note.swara.rstrip("+-") in allowed:
            continue
        semitones = note.midi - result.tonic_midi
        best_swara, best_octave, best_delta = "", 0, 1e9
        for octave in range(-4, 5):
            for swara in allowed:
                delta = semitones - (SWARA_SEMITONES.get(swara, 0) + 12 * octave)
                if abs(delta) < abs(best_delta):
                    best_swara, best_octave, best_delta = swara, octave, delta
        if best_swara and abs(best_delta) * 100.0 <= tolerance_cents:
            marks = "+" * best_octave if best_octave > 0 else "-" * (-best_octave)
            note.swara = best_swara + marks
            note.cents_deviation = best_delta * 100.0
    return result


def analyse_file(path: Path, raaga: Optional[Raaga] = None,
                 tonic_hint_midi: Optional[float] = None) -> AnalysisResult:
    audio, sr = load_audio(Path(path))
    result = analyse(audio, sr, raaga, tonic_hint_midi)
    log.info("analysed %s: %s", Path(path).name, result.summary())
    return result


def phrase_features(phrase: AnalysedPhrase) -> Dict[str, object]:
    """Compact feature description used for storage and comparison."""
    midi = phrase.midi
    return {
        "length": len(phrase.notes),
        "span": (max(midi) - min(midi)) if midi else 0,
        "shape": phrase.shape(),
        "contour": phrase.contour(),
        "duration": round(phrase.end - phrase.start, 3),
        "ends_on": phrase.swaras[-1] if phrase.swaras else "",
        "starts_on": phrase.swaras[0] if phrase.swaras else "",
    }
