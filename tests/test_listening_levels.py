"""The common-level fallback, exercised on both branches without a model.

Two review passes found this same area wrong in two different ways. First the
guard set `peak_cap_exceeded` and wrote the unadjusted samples anyway - a cap
that was announced and never applied. Then the fix lowered only the candidate
and still pointed the comparison at the original -18 dBFS C, which would have
put two files at different levels beside each other and called them matched.

Neither reached a listener, because the real candidate peaks at 0.373 and never
takes the capped branch at all. That is exactly why it needs a fixture: the
branch that has never run is the one nobody has checked.

Generated audio only. No model, no inference, no GPU, and no existing render is
read for its content or written to.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trials import current_lyrics_morning_settings as step2   # noqa: E402

RAIL_P, RAIL_N = 1.0 - 2.0 ** -23, -1.0


def tone(seconds=1.0, hz=220.0, peak=0.5, rate=step2.SR, crest=1.0):
    """A tone whose crest factor can be pushed up, so the cap can be reached."""
    t = np.arange(int(seconds * rate)) / rate
    wave = np.sin(2 * np.pi * hz * t)
    if crest > 1.0:
        # A short spike raises peak without moving RMS much, which is what makes
        # a signal unable to reach a target level under a peak cap.
        wave[len(wave) // 2] = crest
    return wave / max(float(np.max(np.abs(wave))), 1e-12) * peak


class ListeningLevelTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name)
        # A stand-in for C's preserved raw, so nothing real is touched.
        self.c_raw = self.out / "c-raw.wav"
        sf.write(str(self.c_raw), tone(1.0, 180.0, 0.6), step2.SR, subtype="PCM_24")
        self.c_listening = self.out / "c-listening.wav"
        sf.write(str(self.c_listening), tone(1.0, 180.0, 0.3), step2.SR,
                 subtype="PCM_24")
        self._saved = (step2.C_RAW, step2.C_LISTENING, dict(step2.EXPECTED))
        step2.C_RAW = self.c_raw
        step2.C_LISTENING = self.c_listening
        step2.EXPECTED = {self.c_listening: step2.sha(self.c_listening)}

    def tearDown(self):
        step2.C_RAW, step2.C_LISTENING, step2.EXPECTED = self._saved

    def measure(self, path):
        audio, _ = sf.read(str(path), dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return {"rms_dbfs": 20 * np.log10(float(np.sqrt(np.mean(audio ** 2)))),
                "peak": float(np.max(np.abs(audio))),
                "rails": int(np.sum(audio >= RAIL_P) + np.sum(audio <= RAIL_N))}

    # -- the ordinary branch, which is what the real candidate took ----------
    def test_below_the_cap_uses_the_requested_level_and_the_preserved_c(self):
        record = step2.prepare_listening(tone(1.0, 220.0, 0.4), self.out, sf, np)
        self.assertFalse(record["peak_cap_exceeded"])
        self.assertEqual(record["applied_target_rms_dbfs"], step2.LISTEN_TARGET_DBFS)
        self.assertAlmostEqual(record["achieved_rms_dbfs"], -18.0, places=2)
        self.assertEqual(record["compare_with"], str(self.c_listening),
                         "with no cap, the preserved C copy is the comparison")
        self.assertIn("minus18p0", Path(record["path"]).name)
        self.assertNotIn("compare_with_detail", record)

    # -- the branch nobody had run ------------------------------------------
    def test_above_the_cap_lowers_BOTH_and_says_so(self):
        """The correction. A lowered candidate beside an unlowered C is not a
        level-matched comparison, whatever the record calls it."""
        record = step2.prepare_listening(tone(1.0, 220.0, 0.9, crest=40.0),
                                         self.out, sf, np)
        self.assertTrue(record["peak_cap_exceeded"])
        applied = record["applied_target_rms_dbfs"]
        self.assertLess(applied, step2.LISTEN_TARGET_DBFS)

        # A new C copy exists, at that same level, and is what we compare with.
        self.assertIn("compare_with_detail", record)
        c_copy = Path(record["compare_with"])
        self.assertTrue(c_copy.is_file())
        self.assertNotEqual(c_copy, self.c_listening)
        self.assertEqual(record["compare_with_detail"]["applied_target_rms_dbfs"],
                         applied)

        candidate = self.measure(record["path"])
        comparison = self.measure(c_copy)
        self.assertAlmostEqual(candidate["rms_dbfs"], applied, places=2)
        self.assertAlmostEqual(comparison["rms_dbfs"], applied, places=2)
        self.assertAlmostEqual(candidate["rms_dbfs"], comparison["rms_dbfs"],
                               places=2, msg="both copies must sit at one level")

    def test_the_capped_candidate_actually_respects_the_cap(self):
        record = step2.prepare_listening(tone(1.0, 220.0, 0.9, crest=40.0),
                                         self.out, sf, np)
        self.assertLessEqual(record["achieved_peak"], step2.PEAK_CAP + 1e-6,
                             "announcing a cap is not applying one")
        self.assertEqual(record["pcm24_samples_at_positive_rail"], 0)
        self.assertEqual(record["pcm24_samples_at_negative_rail"], 0)

    def test_the_filenames_state_the_level_they_are(self):
        record = step2.prepare_listening(tone(1.0, 220.0, 0.9, crest=40.0),
                                         self.out, sf, np)
        applied = record["applied_target_rms_dbfs"]
        expected = step2.level_name(applied)
        self.assertIn(expected, Path(record["path"]).name)
        self.assertIn(expected, Path(record["compare_with"]).name)
        self.assertNotIn("minus18p0", Path(record["path"]).name,
                         "a file at another level must not be named -18")

    def test_c_is_never_rewritten(self):
        before = (step2.sha(self.c_listening), step2.sha(self.c_raw))
        step2.prepare_listening(tone(1.0, 220.0, 0.9, crest=40.0), self.out, sf, np)
        self.assertEqual((step2.sha(self.c_listening), step2.sha(self.c_raw)),
                         before, "C's own files are inputs, never outputs")

    def test_the_lower_c_copy_comes_from_c_raw_not_from_its_listening_copy(self):
        """Rescaling an already-normalised file would stack two gains."""
        record = step2.prepare_listening(tone(1.0, 220.0, 0.9, crest=40.0),
                                         self.out, sf, np)
        self.assertEqual(record["compare_with_detail"]["from"], str(self.c_raw))
        self.assertEqual(record["compare_with_detail"]["processing"],
                         "one scalar gain only")

    def test_gain_is_scalar_only(self):
        source = tone(1.0, 220.0, 0.4)
        record = step2.prepare_listening(source, self.out, sf, np)
        written, _ = sf.read(record["path"], dtype="float64")
        expected = source * 10 ** (record["gain_db"] / 20)
        self.assertLess(float(np.max(np.abs(written - expected))), 2.0 ** -23,
                        "gain only means one multiply, within a 24-bit step")


if __name__ == "__main__":
    unittest.main()
