"""Exercise the headroom pass end to end with a stand-in conversion.

The point is the bookkeeping around the model call, not the model: a prelaunch
review found that `measure_pcm` returned a `path` key while every call site also
passed `path=`, which would have raised `TypeError` *after* the conversion had
already run. A single model launch is expensive and this pass is authorized once,
so the runner is exercised first with a callable that costs nothing.

No model, GPU, network or training.
"""
import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trials import headroom_pass                                  # noqa: E402


class StandInConversion:
    """Writes a plausible output where the real adapter would, and records the
    same fields. Deliberately produces a clipped result, because the interesting
    path is the one where the pass has to report that it did not help."""

    def __init__(self, peak=1.0):
        self.peak = peak

    def __call__(self, guide, reference, directory, record):
        audio, rate = sf.read(str(guide), dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        scaled = audio / max(float(np.max(np.abs(audio))), 1e-9) * self.peak
        out = Path(directory) / "conversion-output"
        out.mkdir(parents=True, exist_ok=True)
        destination = out / "vc_stand-in.wav"
        sf.write(str(destination), scaled, rate, subtype="PCM_24")
        record.settings(backend="stand-in", device="cpu", precision="float32",
                        seed=headroom_pass.SEED, training=False,
                        **headroom_pass.SETTINGS)
        return destination


class HeadroomPassTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.renders = self.root / "renders"
        # A fixture baseline shaped exactly like the real one.
        self.baseline = self.renders / "r-fixture-baseline"
        (self.baseline / "conversion-output").mkdir(parents=True)
        (self.baseline / "listening").mkdir(parents=True)
        shutil.copy2(headroom_pass.GUIDE, self.baseline / "guide.wav")
        for name in ("listening/B-converted-minus18.wav",):
            shutil.copy2(headroom_pass.LISTENING_B, self.baseline / name)
        real = json.loads((headroom_pass.BASELINE / "render.json").read_text(
            encoding="utf-8"))
        (self.baseline / "render.json").write_text(json.dumps(real), encoding="utf-8")
        self._original = (headroom_pass.BASELINE, headroom_pass.BASELINE_ID,
                          headroom_pass.GUIDE, headroom_pass.LISTENING_B)
        headroom_pass.GUIDE = self.baseline / "guide.wav"
        headroom_pass.LISTENING_B = self.baseline / "listening" / "B-converted-minus18.wav"

    def tearDown(self):
        (headroom_pass.BASELINE, headroom_pass.BASELINE_ID,
         headroom_pass.GUIDE, headroom_pass.LISTENING_B) = self._original

    def run_pass(self, peak=1.0):
        headroom_pass.main(lambda: StandInConversion(peak),
                           out_root=self.renders, baseline=self.baseline)
        made = [p for p in self.renders.iterdir()
                if p.is_dir() and p.name != self.baseline.name]
        self.assertEqual(len(made), 1)
        return json.loads((made[0] / "render.json").read_text(encoding="utf-8"))

    def test_the_whole_pass_completes_and_records_every_field(self):
        """The call-binding failure would have raised here, after conversion."""
        record = self.run_pass()
        self.assertEqual(record["status"], "rendered_not_accepted")
        self.assertEqual(record["parent_render_id"], self.baseline.name)
        for name in ("baseline_guide", "attenuated_guide", "reference"):
            self.assertIn(name, record["inputs"], name)
        self.assertIn("converted", record["outputs"])
        self.assertIn("listening_converted", record["outputs"])
        self.assertIn("level_matched_comparison", record)
        self.assertIn("feedback_binding", record)

    def test_the_input_is_exactly_half_and_otherwise_unchanged(self):
        record = self.run_pass()
        before = record["inputs"]["baseline_guide"]
        after = record["inputs"]["attenuated_guide"]
        self.assertEqual(before["frames"], after["frames"])
        self.assertEqual(before["sample_rate"], after["sample_rate"])
        self.assertAlmostEqual(after["peak"] / before["peak"], 0.5, places=4)
        self.assertAlmostEqual(after["rms_dbfs"] - before["rms_dbfs"], -6.02, places=1)
        self.assertEqual(record["stages"]["attenuation"]["gain_linear"], 0.5)

    def test_the_control_change_and_its_parent_are_recorded(self):
        record = self.run_pass()
        change = record["control_change"]
        self.assertEqual((change["before"], change["after"]), (1.0, 0.5))
        self.assertAlmostEqual(change["decibels"], -6.021, places=2)
        self.assertIn("not attenuation of the already-clipped output",
                      change["not_this"])
        self.assertEqual(record["feedback_binding"]["parent_render"],
                         self.baseline.name)

    def test_a_still_clipped_result_is_reported_and_not_smoothed(self):
        record = self.run_pass(peak=1.0)
        out = record["outputs"]["converted"]
        # Either rail counts as clipped. Which one a normalised signal reaches
        # depends on whether its largest excursion happens to be positive.
        self.assertGreater(out["samples_at_positive_rail"]
                           + out["samples_at_negative_rail"], 0)
        self.assertTrue(any("Still clipped" in w for w in record["warnings"]))
        self.assertTrue(any("does not guarantee" in w for w in record["warnings"]))

    def test_a_clean_result_reports_no_rails(self):
        record = self.run_pass(peak=0.5)
        out = record["outputs"]["converted"]
        self.assertEqual(out["samples_at_positive_rail"], 0)
        self.assertEqual(out["samples_at_negative_rail"], 0)
        self.assertEqual(record["warnings"], [])

    def test_rails_are_counted_at_the_true_24_bit_extremes(self):
        """0.999999 is a looser question than 'clipped', and answers differently."""
        path = self.root / "edge.wav"
        audio = np.zeros(64)
        audio[0] = 1.0                     # positive rail
        audio[1] = -1.0                    # negative rail
        audio[2] = 0.9999995               # inside the rail, above 0.999999
        sf.write(str(path), audio, 44100, subtype="PCM_24")
        measured = headroom_pass.measure_pcm(path)
        self.assertNotIn("path", measured,
                         "measure_pcm must not return a key its callers also pass")
        self.assertEqual(measured["samples_at_positive_rail"], 1)
        self.assertEqual(measured["samples_at_negative_rail"], 1)
        self.assertGreaterEqual(measured["diagnostic_near_rail_positive"], 2)
        self.assertEqual(measured["rail_definition"]["negative"], -1.0)

    def test_a_changed_guide_stops_the_pass(self):
        sf.write(str(headroom_pass.GUIDE), np.zeros(1000), 44100, subtype="PCM_24")
        with self.assertRaises(headroom_pass.PassError) as raised:
            self.run_pass()
        self.assertIn("not the baseline guide", str(raised.exception))

    def test_changed_settings_stop_the_pass(self):
        stored = json.loads((self.baseline / "render.json").read_text(encoding="utf-8"))
        stored["settings"]["inference_cfg_rate"] = 0.3
        (self.baseline / "render.json").write_text(json.dumps(stored), encoding="utf-8")
        with self.assertRaises(headroom_pass.PassError) as raised:
            self.run_pass()
        self.assertIn("inference_cfg_rate", str(raised.exception))

    def test_a_changed_input_hash_stops_the_pass(self):
        stored = json.loads((self.baseline / "render.json").read_text(encoding="utf-8"))
        stored["inputs"]["checkpoint"]["sha256"] = "f" * 64
        (self.baseline / "render.json").write_text(json.dumps(stored), encoding="utf-8")
        with self.assertRaises(headroom_pass.PassError) as raised:
            self.run_pass()
        self.assertIn("same bytes on both sides", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
