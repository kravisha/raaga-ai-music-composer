"""Invariants for the steady-region detector, from the review's own cases.

Both of these were failures, not hypotheticals. A stationary harmonic tone was
rejected as unsteady because a raw log spectrum moves with the analysis
window's phase, and the returned end ran a hop past the audio because the run's
end index is exclusive.

No model, no audio files, no rendering — generated signals only.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trials.shape_to_score import steady_region              # noqa: E402

RATE = 16000


def pitch_track(seconds: float, hz: float, step: float = 0.005):
    times = np.arange(0, seconds, step)
    return times, np.full(len(times), hz)


def harmonic(seconds: float, hz: float, rate: int = RATE) -> np.ndarray:
    """The review's signal: a stationary harmonic tone, not a pure sine."""
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * hz * t)
            + 0.5 * np.sin(2 * np.pi * 2 * hz * t)
            + 0.25 * np.sin(2 * np.pi * 3 * hz * t))


class SteadyRegionTests(unittest.TestCase):
    def test_a_stationary_harmonic_tone_is_steady(self):
        """The reported failure. A raw log spectrum moves with window phase, so
        this was rejected; band energies do not."""
        piece = harmonic(0.4, 250.0)
        times, freqs = pitch_track(0.4, 250.0)
        start, end, how = steady_region(np, piece, RATE, times, freqs, 0.0, 0.03)
        self.assertIsNotNone(start, f"stationary harmonic rejected: {how}")
        self.assertGreater(end - start, 0.25,
                           "most of a stationary tone should read as steady")

    def test_a_pure_sine_is_still_steady(self):
        t = np.arange(int(0.4 * RATE)) / RATE
        piece = np.sin(2 * np.pi * 250.0 * t)
        times, freqs = pitch_track(0.4, 250.0)
        start, end, _ = steady_region(np, piece, RATE, times, freqs, 0.0, 0.03)
        self.assertIsNotNone(start)
        self.assertGreater(end - start, 0.25)

    def test_the_end_never_runs_past_the_audio(self):
        """`best_run[1]` is exclusive; using it put the end a hop beyond."""
        for seconds in (0.20, 0.40, 0.55):
            with self.subTest(seconds=seconds):
                piece = harmonic(seconds, 220.0)
                times, freqs = pitch_track(seconds, 220.0)
                start, end, _ = steady_region(np, piece, RATE, times, freqs,
                                              0.0, 0.03)
                self.assertIsNotNone(start)
                self.assertLessEqual(end, seconds + 1e-9,
                                     "the nucleus must end inside the audio")
                self.assertGreaterEqual(start, 0.0)

    def test_the_offset_is_honoured(self):
        piece = harmonic(0.4, 250.0)
        times, freqs = pitch_track(2.4, 250.0)
        start, end, _ = steady_region(np, piece, RATE, times, freqs, 2.0, 0.03)
        self.assertIsNotNone(start)
        self.assertGreaterEqual(start, 2.0)
        self.assertLessEqual(end, 2.4 + 1e-9)

    def test_a_quiet_steady_tail_is_not_taken_for_the_vowel(self):
        """A sustained nasal is steady too, and quieter. The loud run wins —
        though this is an energy heuristic, not a phonetic decision."""
        loud = harmonic(0.20, 250.0)
        quiet = harmonic(0.30, 250.0) * 0.12
        piece = np.concatenate([loud, quiet])
        times, freqs = pitch_track(0.5, 250.0)
        start, end, how = steady_region(np, piece, RATE, times, freqs, 0.0, 0.03)
        self.assertIsNotNone(start, how)
        self.assertLess(start, 0.20, "the chosen run should sit in the loud part")

    def test_a_moving_signal_is_not_steady(self):
        t = np.arange(int(0.4 * RATE)) / RATE
        rising = np.linspace(200, 600, len(t))
        sweep = np.sin(2 * np.pi * np.cumsum(rising) / RATE)
        # The pitch track has to sweep too. Handing a flat track for a sweeping
        # signal was my test being unrealistic, not the detector being wrong.
        times = np.arange(0, 0.4, 0.005)
        freqs = np.interp(times, t, rising)
        start, _, how = steady_region(np, sweep, RATE, times, freqs, 0.0, 0.20)
        self.assertIsNone(start, f"a sweep should not read as steady: {how}")

    def test_the_result_is_labelled_an_estimate(self):
        piece = harmonic(0.4, 250.0)
        times, freqs = pitch_track(0.4, 250.0)
        _, _, how = steady_region(np, piece, RATE, times, freqs, 0.0, 0.03)
        self.assertIn("ESTIMATE", how)
        self.assertIn("not", how.lower())


if __name__ == "__main__":
    unittest.main()
