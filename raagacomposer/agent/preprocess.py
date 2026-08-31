"""Preparing a real recording for the ears.

The analysis pipeline in :mod:`analysis` assumes what a rendered exercise gives
it: one voice, sounding alone, pitched from the first sample to the last.  A
recording of an actual lesson is not that.  It is a person talking, over a
tanpura or a shruti box that never stops, occasionally singing.  Fed to the
autocorrelation tracker unchanged, the drone competes with the voice for every
frame and the spoken passages arrive as melody.

This module sits *before* analysis and does two things:

    drone           find the sustained pitch, take Sa from it, notch it out
    sung spans      silence the stretches that are speech rather than singing

Both are deliberately conservative.  Where either is unsure it says so and
leaves the audio alone, because a phrase that is never learned costs one
phrase, and a wrong phrase written into permanent memory costs the confidence
of everything downstream that reads it.

The drone is treated as a gift rather than a nuisance.  A tanpura exists to
declare Sa; when we can find it, its fundamental is a far better tonic than
anything estimation from the melody could produce, and it is handed to
:func:`analysis.analyse` as a fixed tonic instead of being guessed at.

Nothing here splices the audio.  Rejected stretches are silenced in place, so
every timestamp still means what it meant and the silence becomes a phrase
boundary that :func:`analysis.segment_phrases` already knows how to read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.logging_setup import get_logger
from ..music.theory import freq_to_midi
from . import analysis

log = get_logger("agent.preprocess")

PREPROCESS_VERSION = "drone-notch-speech-gate-1"

# -- drone ----------------------------------------------------------------
# A drone does not move, so frequency resolution is worth far more here than
# time resolution: a long window costs nothing and halves the bin width.
STFT_N = 4096
STFT_HOP = 1024
DRONE_F_MIN = 80.0          # below a bass tanpura's lower Sa
DRONE_F_MAX = 400.0         # above a high shruti box Sa
# A long held note in an alapana is stationary too, and notching it out would
# remove the very thing worth learning from.  Two things tell a drone from a
# held note: a drone never stops for the whole recording, and it sounds a
# chord of partials - Sa with its Pa and its octave - where a sung note brings
# only itself.  Held notes measure around 0.5 and one or two partials; a real
# tanpura measures above 0.89 with four.
DRONE_MIN_STATIONARITY = 0.70
DRONE_MIN_COVERAGE = 0.6    # it has to be sounding most of the time
DRONE_MIN_PARTIALS = 3
NOTCH_Q = 26.0              # narrow: a drone partial is one pitch, not a band

# Partials a tanpura or shruti box actually puts out, as ratios to Sa.  The Pa
# below (2/3) is the tanpura's first string; the rest are Sa's own harmonics
# and the Pa above.
DRONE_RATIOS = (2.0 / 3.0, 1.0, 1.5, 2.0, 3.0, 4.0)

# -- sung-note gate -------------------------------------------------------
GATE_WINDOW = 2.0           # seconds of audio judged at a time
GATE_HOP = 0.5
# Speech is not uniformly unmusical: at a turning point in the pitch contour
# it really does hold still for a moment, and a short enough window sees only
# that moment.  Two seconds is long enough to contain the glide either side.
# A stretch shorter than this is in any case not a phrase worth learning, so
# runs below it are dropped rather than trusted.
MIN_SUNG_RUN = 3.0
PLATEAU_CENTS = 50.0        # how far pitch may wander and still be "held"
PLATEAU_MIN_SECONDS = 0.12  # ... and for how long before it counts as a note
# Gamaka swings a note by a semitone or more, so the raw contour of good
# singing is not flat at all.  Taking a running median over rather more than
# one oscillation leaves the centre the ornament is decorating: a kampita
# medians to the note it belongs to, while a spoken glide medians to a glide,
# because a median flattens oscillation but not a slope.
PLATEAU_SMOOTH_SECONDS = 0.18
GATE_THRESHOLD = 0.50
FADE_SECONDS = 0.02         # so silencing a span does not click


@dataclass
class DroneEstimate:
    """A sustained pitch running under the whole recording."""

    hz: float = 0.0
    confidence: float = 0.0
    stationarity: float = 0.0
    coverage: float = 0.0
    partials: List[float] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.hz > 0.0 and self.confidence > 0.0

    @property
    def midi(self) -> float:
        return freq_to_midi(self.hz) if self.hz > 0 else 0.0

    def summary(self) -> str:
        if not self.found:
            return "no drone found"
        return (f"drone at {self.hz:.1f} Hz (MIDI {self.midi:.1f}), "
                f"{len(self.partials)} partial(s), "
                f"confidence {self.confidence:.2f}")


@dataclass
class Span:
    """A stretch of the recording and how much it sounds like singing."""

    start: float
    end: float
    music_likelihood: float
    voiced_fraction: float = 0.0
    plateau_fraction: float = 0.0
    #: Set once the run filter has had its say; None means "score alone".
    verdict: Optional[bool] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def is_sung(self) -> bool:
        if self.verdict is not None:
            return self.verdict
        return self.music_likelihood >= GATE_THRESHOLD


@dataclass
class PreparedAudio:
    """What the ears should listen to, and what was taken out of the way."""

    audio: np.ndarray
    sample_rate: int
    drone: DroneEstimate = field(default_factory=DroneEstimate)
    spans: List[Span] = field(default_factory=list)
    kept_seconds: float = 0.0
    silenced_seconds: float = 0.0
    drone_removed: bool = False
    version: str = PREPROCESS_VERSION
    warnings: List[str] = field(default_factory=list)

    @property
    def tonic_midi(self) -> Optional[float]:
        """The tonic the drone declares, if it declared one clearly enough."""
        return self.drone.midi if self.drone.confidence >= 0.5 else None

    @property
    def sung_fraction(self) -> float:
        total = self.kept_seconds + self.silenced_seconds
        return self.kept_seconds / total if total > 0 else 0.0

    def summary(self) -> str:
        parts = [self.drone.summary()]
        if self.silenced_seconds > 0:
            parts.append(f"{self.silenced_seconds:.1f}s silenced as speech "
                         f"({self.sung_fraction:.0%} kept)")
        else:
            parts.append("nothing silenced")
        return "; ".join(parts)


# --------------------------------------------------------------------------
# spectra
# --------------------------------------------------------------------------
def _magnitude_spectrogram(audio: np.ndarray, sr: int, n_fft: int = STFT_N,
                           hop: int = STFT_HOP
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Plain STFT magnitude. Returns (magnitude[bin, frame], bin frequencies)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < n_fft:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32), \
            np.fft.rfftfreq(n_fft, 1.0 / sr)
    window = np.hanning(n_fft).astype(np.float32)
    count = 1 + (len(audio) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(count, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0]))
    spectrum = np.fft.rfft(frames * window, axis=1)
    return np.abs(spectrum).T.astype(np.float32), np.fft.rfftfreq(n_fft, 1.0 / sr)


def _stationary_spectrum(magnitude: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Separate what is always there from what comes and goes.

    A drone occupies the same bins in nearly every frame, so the *median* over
    time of those bins is high.  A sung note passes through a bin and leaves,
    so its median is low even where its mean is not.  The ratio of the two is
    therefore a direct measure of how stationary a bin is, and that is exactly
    what distinguishes an accompaniment from a melody.
    """
    if magnitude.shape[1] == 0:
        empty = np.zeros(magnitude.shape[0], dtype=np.float32)
        return empty, empty
    median = np.median(magnitude, axis=1).astype(np.float32)
    mean = magnitude.mean(axis=1).astype(np.float32)
    stationarity = median / (mean + 1e-9)
    return median, np.clip(stationarity, 0.0, 1.0).astype(np.float32)


def _interpolated_peak(spectrum: np.ndarray, index: int,
                       resolution: float) -> float:
    """Refine a spectral peak to a fraction of a bin.

    A bin is about 5 Hz wide, which around a low Sa is most of a semitone -
    far too coarse for a tonic, since every swara downstream is measured from
    it.  Fitting a parabola through the peak and its neighbours, in the log
    domain where a windowed sinusoid's main lobe is very nearly quadratic,
    recovers the true frequency to a small fraction of a bin.
    """
    if not (1 <= index < len(spectrum) - 1):
        return index * resolution
    a, b, c = (math.log(max(float(spectrum[index + offset]), 1e-12))
               for offset in (-1, 0, 1))
    denominator = a - 2.0 * b + c
    if abs(denominator) < 1e-12:
        return index * resolution
    shift = 0.5 * (a - c) / denominator
    return (index + float(np.clip(shift, -0.5, 0.5))) * resolution


def _refine_fundamental(spectrum: np.ndarray, freq: float, resolution: float,
                        ratios: Sequence[float]) -> float:
    """Sharpen a drone's fundamental using every partial that is present.

    A partial at four times the fundamental carries four times the absolute
    frequency error for the same relative error, so dividing its own refined
    peak back down gives a far better estimate of Sa than the fundamental's
    bin can.  The partials are combined weighted by ratio for that reason.
    """
    estimates: List[Tuple[float, float]] = []
    for ratio in ratios:
        target = freq * ratio
        index = int(round(target / resolution))
        if not (1 <= index < len(spectrum) - 1):
            continue
        # Only trust a partial that is a peak in its own right.
        if spectrum[index] < spectrum[index - 1] or spectrum[index] < spectrum[index + 1]:
            continue
        refined = _interpolated_peak(spectrum, index, resolution)
        if refined <= 0:
            continue
        candidate = refined / ratio
        if abs(candidate - freq) > freq * 0.03:
            continue
        estimates.append((candidate, ratio))
    if not estimates:
        return freq
    weights = np.array([r for _, r in estimates], dtype=np.float64)
    values = np.array([v for v, _ in estimates], dtype=np.float64)
    return float(np.sum(values * weights) / np.sum(weights))


def _coverage(magnitude: np.ndarray, bin_index: int, floor: float) -> float:
    """The share of frames in which this bin is actually sounding."""
    if magnitude.shape[1] == 0 or not (0 <= bin_index < magnitude.shape[0]):
        return 0.0
    return float(np.mean(magnitude[bin_index] > floor))


# --------------------------------------------------------------------------
# drone
# --------------------------------------------------------------------------
def detect_drone(audio: np.ndarray, sr: int,
                 f_min: float = DRONE_F_MIN,
                 f_max: float = DRONE_F_MAX) -> DroneEstimate:
    """Find the sustained pitch under a recording, if there is one.

    Candidates are scored by summing the stationary spectrum at the ratios a
    tanpura or shruti box actually sounds, so the fundamental that best
    explains the whole stationary picture wins rather than merely the loudest
    steady bin - which would otherwise pick the Pa string or an octave.
    """
    estimate = DroneEstimate()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < STFT_N * 2:
        return estimate

    magnitude, freqs = _magnitude_spectrogram(audio, sr)
    if magnitude.shape[1] < 4:
        return estimate
    median, stationarity = _stationary_spectrum(magnitude)
    if not np.any(median > 0):
        return estimate

    # Only bins that are both loud and steady can belong to a drone.
    steady = median * stationarity
    if steady.max() <= 0:
        return estimate
    steady = steady / steady.max()

    resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    lo = max(1, int(math.floor(f_min / resolution)))
    hi = min(len(freqs) - 2, int(math.ceil(f_max / resolution)))
    if hi <= lo:
        return estimate

    def support(freq: float) -> float:
        """How much stationary energy sits at this pitch and its partials."""
        total = 0.0
        for ratio in DRONE_RATIOS:
            target = freq * ratio
            if target <= 0 or target >= freqs[-1]:
                continue
            index = int(round(target / resolution))
            window = steady[max(0, index - 1):index + 2]
            if len(window):
                weight = 1.6 if ratio == 1.0 else 1.0
                total += weight * float(window.max())
        return total

    best_freq, best_support = 0.0, 0.0
    for index in range(lo, hi + 1):
        # Only consider local maxima: a drone is a peak, not a shoulder.
        if steady[index] < steady[index - 1] or steady[index] < steady[index + 1]:
            continue
        if steady[index] < 0.05:
            continue
        freq = float(freqs[index])
        value = support(freq)
        if value > best_support:
            best_support, best_freq = value, freq

    if best_freq <= 0:
        return estimate

    # The winning bin names the drone to within about half a semitone.  Every
    # swara downstream is measured from this pitch, so refine it before it is
    # used as anything.
    best_freq = _refine_fundamental(median, best_freq, resolution, DRONE_RATIOS)

    index = int(round(best_freq / resolution))
    floor = float(np.median(magnitude)) * 2.0
    estimate.hz = best_freq
    estimate.stationarity = float(stationarity[index])
    estimate.coverage = _coverage(magnitude, index, floor)

    # Which partials are really present - those are the ones worth notching.
    partials: List[float] = []
    for ratio in DRONE_RATIOS:
        target = best_freq * ratio
        if target <= 0 or target >= min(freqs[-1], sr / 2.0 * 0.95):
            continue
        partial_index = int(round(target / resolution))
        if partial_index >= len(steady):
            continue
        if steady[max(0, partial_index - 1):partial_index + 2].max() >= 0.05:
            partials.append(float(target))
    estimate.partials = partials

    if (estimate.stationarity < DRONE_MIN_STATIONARITY
            or estimate.coverage < DRONE_MIN_COVERAGE
            or len(partials) < DRONE_MIN_PARTIALS):
        log.debug("a steady pitch at %.1f Hz was not a drone (stationarity "
                  "%.2f, coverage %.2f, %d partial(s)) - most likely a held "
                  "note, which is left alone",
                  best_freq, estimate.stationarity, estimate.coverage,
                  len(partials))
        estimate.confidence = 0.0
        return estimate

    # Confidence: how steady it is, how much of the time it sounds, and how
    # much of a tanpura's partial pattern it actually shows.
    pattern = min(1.0, len(partials) / 4.0)
    estimate.confidence = round(
        float(np.clip(0.45 * estimate.stationarity + 0.30 * estimate.coverage
                      + 0.25 * pattern, 0.0, 1.0)), 3)
    return estimate


def suppress_drone(audio: np.ndarray, sr: int, drone: DroneEstimate,
                   q: float = NOTCH_Q) -> np.ndarray:
    """Notch a drone's partials out, leaving the melody either side of them."""
    if not drone.found or not drone.partials:
        return audio
    try:
        from scipy.signal import filtfilt, iirnotch
    except Exception as exc:  # noqa: BLE001 - scipy is a dependency, not a given
        log.warning("no scipy: the drone cannot be notched out (%s)", exc)
        return audio

    filtered = np.asarray(audio, dtype=np.float64).reshape(-1).copy()
    nyquist = sr / 2.0
    for partial in drone.partials:
        if not (0 < partial < nyquist * 0.98):
            continue
        try:
            b, a = iirnotch(partial / nyquist, q)
            filtered = filtfilt(b, a, filtered)
        except Exception as exc:  # noqa: BLE001
            log.debug("could not notch %.1f Hz: %s", partial, exc)
    peak = float(np.abs(filtered).max())
    if peak > 1e-9:
        filtered = filtered / peak
    return filtered.astype(np.float32)


# --------------------------------------------------------------------------
# telling singing from speech
# --------------------------------------------------------------------------
def _median_filter(values: np.ndarray, width: int) -> np.ndarray:
    """Running median - it flattens an oscillation but leaves a slope sloping."""
    if width < 3 or len(values) < width:
        return values
    if width % 2 == 0:
        width += 1
    padded = np.pad(values, (width // 2, width // 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _plateau_fraction(f0: np.ndarray, hop_seconds: float) -> float:
    """How much of the voiced pitch is *held* rather than sliding.

    This is the difference that matters.  A sung note settles on a pitch and
    stays there; where it is ornamented it oscillates *around* that pitch
    rather than leaving it, which the running median above reduces back to the
    note.  Speech never settles: its pitch glides from the start of a syllable
    to the end and moves on, and a median leaves a glide gliding.  So the share
    of voiced time spent inside a plateau separates the two without needing to
    know any language, and without a model.
    """
    voiced = f0 > 0
    if not np.any(voiced):
        return 0.0
    cents = np.zeros(len(f0), dtype=np.float64)
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / 440.0)
    cents = _median_filter(
        cents, int(round(PLATEAU_SMOOTH_SECONDS / max(1e-6, hop_seconds))))

    min_frames = max(2, int(round(PLATEAU_MIN_SECONDS / max(1e-6, hop_seconds))))
    held = 0
    index = 0
    total_voiced = int(np.count_nonzero(voiced))
    while index < len(f0):
        if not voiced[index]:
            index += 1
            continue
        end = index + 1
        while end < len(f0) and voiced[end]:
            run = cents[index:end + 1]
            if run.max() - run.min() > PLATEAU_CENTS:
                break
            end += 1
        length = end - index
        if length >= min_frames:
            held += length
        index = max(end, index + 1)
    return float(held / total_voiced) if total_voiced else 0.0


def score_window(f0: np.ndarray, hop_seconds: float) -> Tuple[float, float, float]:
    """Judge one window. Returns (likelihood, voiced fraction, plateau share)."""
    if len(f0) == 0:
        return 0.0, 0.0, 0.0
    voiced_fraction = float(np.mean(f0 > 0))
    plateau = _plateau_fraction(f0, hop_seconds)
    # Held pitch decides it.  Voicing only discounts the verdict, because a
    # window that is barely voiced has too little evidence either way - it
    # cannot promote a window that holds nothing into singing, which a
    # weighted sum of the two would happily do.
    likelihood = plateau * (0.75 + 0.25 * min(1.0, voiced_fraction / 0.6))
    return round(float(np.clip(likelihood, 0.0, 1.0)), 3), \
        round(voiced_fraction, 3), round(plateau, 3)


def sung_spans(audio: np.ndarray, sr: int,
               window_seconds: float = GATE_WINDOW,
               hop_seconds: float = GATE_HOP) -> List[Span]:
    """Walk the recording and judge each window as singing or speech."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = len(audio) / max(1, sr)
    if duration <= 0:
        return []
    if duration <= window_seconds:
        _, f0, _ = analysis.pitch_track(audio, sr)
        likelihood, voiced, plateau = score_window(f0, analysis.HOP_SECONDS)
        return [Span(0.0, duration, likelihood, voiced, plateau)]

    times, f0, _ = analysis.pitch_track(audio, sr)
    if len(times) == 0:
        return []

    spans: List[Span] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        mask = (times >= start) & (times < end)
        likelihood, voiced, plateau = score_window(
            f0[mask], analysis.HOP_SECONDS)
        spans.append(Span(start, end, likelihood, voiced, plateau))
        if end >= duration:
            break
        start += hop_seconds
    return drop_short_runs(spans)


def drop_short_runs(spans: List[Span],
                    minimum: float = MIN_SUNG_RUN) -> List[Span]:
    """Reject sung stretches too short to be a phrase.

    An isolated window scoring as singing in the middle of a minute of talking
    is far more likely to be a moment where the speaker's pitch happened to
    level off than a snatch of song.  Singing lasts; that is what this uses.
    """
    if not spans:
        return spans
    index = 0
    while index < len(spans):
        if not spans[index].is_sung:
            index += 1
            continue
        end = index
        while end + 1 < len(spans) and spans[end + 1].is_sung:
            end += 1
        covered = spans[end].end - spans[index].start
        if covered < minimum:
            for span in spans[index:end + 1]:
                span.verdict = False
        index = end + 1
    return spans


def _silence(audio: np.ndarray, sr: int, start: float, end: float) -> None:
    """Silence a stretch in place, fading either edge so it does not click."""
    first = max(0, int(start * sr))
    last = min(len(audio), int(end * sr))
    if last <= first:
        return
    fade = min(int(FADE_SECONDS * sr), (last - first) // 2)
    if fade > 0:
        audio[first:first + fade] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        audio[last - fade:last] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[first + fade:last - fade] = 0.0
    else:
        audio[first:last] = 0.0


def _merge(spans: Sequence[Span]) -> List[Tuple[float, float]]:
    """Overlapping windows agreeing on speech become one stretch."""
    ranges = [(s.start, s.end) for s in spans if not s.is_sung]
    if not ranges:
        return []
    ranges.sort()
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


# --------------------------------------------------------------------------
# the whole preparation
# --------------------------------------------------------------------------
def prepare(audio: np.ndarray, sr: int, *,
            remove_drone: bool = True,
            gate_speech: bool = True) -> PreparedAudio:
    """Make a real recording safe to hand to the ears.

    The order matters: the drone is notched out first, because the sung-note
    gate works from a pitch track, and an un-notched drone would supply a
    perfectly held pitch through every spoken passage and make the whole
    recording look like singing.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    prepared = PreparedAudio(audio=audio, sample_rate=sr)
    duration = len(audio) / max(1, sr)
    if duration < 0.5:
        prepared.warnings.append("too short to prepare")
        prepared.kept_seconds = duration
        return prepared

    peak = float(np.abs(audio).max())
    if peak < 1e-6:
        prepared.warnings.append("audio is silent")
        return prepared
    audio = audio / peak

    if remove_drone:
        prepared.drone = detect_drone(audio, sr)
        if prepared.drone.found:
            audio = suppress_drone(audio, sr, prepared.drone)
            prepared.drone_removed = True
            log.info("preprocess: %s", prepared.drone.summary())
        else:
            prepared.warnings.append("no drone found; nothing was notched out")

    if gate_speech:
        prepared.spans = sung_spans(audio, sr)
        speech = _merge(prepared.spans)
        if speech and len(speech) == 1 and speech[0][1] - speech[0][0] >= duration - 1e-3:
            # Everything looks like speech.  That is more likely to be a gate
            # that does not suit this recording than a lesson with no singing
            # in it, so leave the audio alone and say so.
            prepared.warnings.append(
                "the whole recording scored as speech; the gate was not applied")
            speech = []
        for start, end in speech:
            _silence(audio, sr, start, end)
            prepared.silenced_seconds += end - start
    prepared.kept_seconds = max(0.0, duration - prepared.silenced_seconds)

    prepared.audio = audio
    return prepared
