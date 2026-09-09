"""Tests for the voice prototype — contract `46e5ca2` (v2).

Fixture-only. The reviewed defects come first, because those are the ones we
know a real implementation gets wrong: Tamil to silence, alignment overflow,
rests sung through, clipping reported as clean, renders overwritten, and
feedback that outlives the bytes it was about.

No model, GPU, network, corpus file or training. The bank and engine are read
where they lie; the carrier tests are marked and skipped if they are absent.
"""
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_prototype import align, feedback, pipeline, render as render_mod  # noqa: E402
from voice_prototype import tune as tune_mod                                 # noqa: E402
from voice_prototype.tune import Event, TuneError                            # noqa: E402

TRIAL = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Tamil-CVVC-Trial-20260908")
BANK_ZIP = TRIAL / "Hanakoloid.zip"
DLL = TRIAL / "runtimes_win-x64_native_worldline.dll"


@dataclass
class Syl:
    """A syllable in the shape the frontend produces."""
    text: str
    onset: str
    vowel: str
    codas: list


def syllables(*names) -> list:
    return [Syl(n, "n", "a", []) for n in names]


class TuneTests(unittest.TestCase):
    def test_direct_controls_give_an_unquantised_curve(self):
        """Fractional pitch survives: the curve is never snapped to a semitone."""
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=60.5)])
        hz = np.array(melody.f0_hz)
        expected = tune_mod.midi_to_hz(60.5)
        self.assertAlmostEqual(float(hz[hz > 0][0]), expected, places=3)
        self.assertNotAlmostEqual(float(hz[hz > 0][0]),
                                  tune_mod.midi_to_hz(60), places=1)

    def test_a_bend_survives_as_written(self):
        """Ornament detail inside an event is not smoothed or quantised."""
        melody = tune_mod.from_events(
            [Event(0.0, 1.0, midi=60, bend=[(0.0, 0.0), (0.5, 100.0), (1.0, 0.0)])])
        hz = np.array(melody.f0_hz)
        base = tune_mod.midi_to_hz(60)
        cents = 1200 * np.log2(hz[hz > 0] / base)
        self.assertAlmostEqual(float(cents.max()), 100.0, delta=2.0)
        self.assertAlmostEqual(float(cents[0]), 0.0, delta=2.0)

    def test_a_rest_is_unvoiced_and_carries_no_pitch(self):
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=60),
                                       Event(1.0, 0.5, midi=None),
                                       Event(1.5, 1.0, midi=62)])
        voiced = np.array(melody.voiced)
        hz = np.array(melody.f0_hz)
        rest = slice(int(1.0 / melody.hop_s) + 1, int(1.5 / melody.hop_s) - 1)
        self.assertFalse(voiced[rest].any(), "a rest must be unvoiced")
        self.assertTrue((hz[rest] == 0).all(), "a rest carries no pitch")

    def test_a_gap_between_events_is_a_rest_too(self):
        """Whether or not anybody wrote the rest down."""
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=60),
                                       Event(2.0, 1.0, midi=62)])
        voiced = np.array(melody.voiced)
        self.assertFalse(voiced[int(1.2 / melody.hop_s)])

    def test_malformed_tunes_are_refused(self):
        for events, why in (
            ([], "empty"),
            ([Event(0.0, 0.0, midi=60)], "no duration"),
            ([Event(-1.0, 1.0, midi=60)], "before zero"),
            ([Event(0.0, float("nan"), midi=60)], "non-finite"),
            ([Event(0.0, 1.0, midi=200)], "out of range"),
            ([Event(0.0, 1.0, midi=60), Event(0.5, 1.0, midi=62)], "overlapping"),
        ):
            with self.subTest(why=why), self.assertRaises(TuneError):
                tune_mod.from_events(events)


class AlignmentTests(unittest.TestCase):
    """The two reviewed defects, and the rules that replace them."""

    def three_notes(self):
        return tune_mod.from_events([Event(0.0, 1.0, midi=60),
                                     Event(1.0, 1.0, midi=62),
                                     Event(2.0, 1.0, midi=64)])

    def test_four_syllables_over_three_notes_do_not_run_past_the_tune(self):
        """The reviewed implementation finished at 4.0 s for a 3 s tune."""
        melody = self.three_notes()
        placed = align.place(syllables("a", "b", "c", "d"), melody)
        self.assertEqual(len(placed), 4)
        self.assertLessEqual(max(p.end_s for p in placed), 3.0 + 1e-6)

    def test_many_syllables_still_end_inside_the_tune(self):
        melody = self.three_notes()
        for count in range(1, 13):
            with self.subTest(count=count):
                placed = align.place(syllables(*[f"s{i}" for i in range(count)]), melody)
                self.assertEqual(len(placed), count)
                self.assertLessEqual(max(p.end_s for p in placed), 3.0 + 1e-6)

    def test_no_syllable_spans_a_rest(self):
        """A held syllable must not bridge silence - that is how a rest gets sung."""
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=60),
                                       Event(1.0, 0.5, midi=None),
                                       Event(1.5, 1.0, midi=62)])
        placed = align.place(syllables("a", "b"), melody)
        self.assertEqual(len({p.phrase for p in placed}), 2,
                         "the rest must split the phrase")
        for item in placed:
            self.assertFalse(item.start_s < 1.5 < item.end_s,
                             f"{item.text} spans the rest")

    def test_syllables_are_shared_across_phrases_not_dumped_in_the_first(self):
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=60),
                                       Event(1.0, 0.5, midi=None),
                                       Event(1.5, 1.0, midi=62),
                                       Event(2.5, 1.0, midi=64)])
        placed = align.place(syllables("a", "b", "c"), melody)
        self.assertEqual(sorted({p.phrase for p in placed}), [0, 1])

    def test_an_empty_lyric_is_refused_not_rendered_as_silence(self):
        with self.assertRaises(align.AlignmentError) as raised:
            align.place([], self.three_notes())
        self.assertIn("successful silence", str(raised.exception))

    def test_a_tune_with_no_sung_events_is_refused(self):
        melody = tune_mod.from_events([Event(0.0, 1.0, midi=None)])
        with self.assertRaises(align.AlignmentError):
            align.place(syllables("a"), melody)

    def test_an_unsayably_dense_lyric_is_refused_rather_than_compressed(self):
        melody = self.three_notes()
        with self.assertRaises(align.AlignmentError) as raised:
            align.place(syllables(*[f"s{i}" for i in range(40)]), melody)
        self.assertIn("unsayable", str(raised.exception))


class RenderRecordTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def wav(self, name, peak=0.5, seconds=0.5, rate=44100):
        path = self.root / name
        audio = np.sin(np.linspace(0, 40 * np.pi, int(rate * seconds))) * peak
        sf.write(str(path), audio, rate, subtype="PCM_24")
        return path

    def test_a_render_directory_is_never_reused(self):
        first, one = render_mod.new_render_dir(self.root)
        second, two = render_mod.new_render_dir(self.root)
        self.assertNotEqual(first, second)
        self.assertNotEqual(one, two)

    def test_the_record_is_written_once(self):
        render_id, directory = render_mod.new_render_dir(self.root)
        record = render_mod.Record(render_id, directory)
        record.finish()
        with self.assertRaises(render_mod.RenderError):
            record.write()

    def test_clipping_is_measured_not_tolerated(self):
        """A tolerance of 1.001 calls a file sitting on the rail clean."""
        path = self.root / "rail.wav"
        audio = np.zeros(1000)
        audio[10:15] = 1.0
        sf.write(str(path), audio, 44100, subtype="PCM_24")
        measured = render_mod.measure_audio(path)
        self.assertEqual(measured["samples_at_rail"], 5)
        self.assertIn("cannot undo it", measured["rail_note"])

    def test_a_clean_file_reports_no_rail_note(self):
        measured = render_mod.measure_audio(self.wav("clean.wav", peak=0.5))
        self.assertEqual(measured["samples_at_rail"], 0)
        self.assertNotIn("rail_note", measured)

    def test_a_listening_copy_is_gain_only_and_says_what_it_did(self):
        source = self.wav("source.wav", peak=0.5)
        copy = render_mod.listening_copy(source, self.root / "copy.wav")
        self.assertEqual(copy["processing"],
                         "gain only; no EQ, limiting, resampling or re-render")
        original, rate = sf.read(str(source), dtype="float64")
        scaled, _ = sf.read(str(copy["to"]), dtype="float64")
        ratio = scaled[np.abs(original) > 1e-6] / original[np.abs(original) > 1e-6]
        self.assertTrue(np.allclose(ratio, ratio[0], atol=1e-3),
                        "gain only means one constant ratio")
        self.assertAlmostEqual(
            float(20 * np.log10(np.sqrt(np.mean(scaled ** 2)))), -18.0, delta=0.1)

    def test_a_listening_copy_does_not_claim_to_repair_a_rail(self):
        path = self.root / "railed.wav"
        audio = np.sin(np.linspace(0, 40 * np.pi, 4410)) * 0.9
        audio[100:103] = 1.0
        sf.write(str(path), audio, 44100, subtype="PCM_24")
        copy = render_mod.listening_copy(path, self.root / "quieter.wav")
        self.assertEqual(copy["source_samples_at_rail"], 3)
        self.assertIn("quieter, not repaired", copy["note"])

    def test_a_copy_is_never_overwritten(self):
        source = self.wav("s.wav")
        render_mod.listening_copy(source, self.root / "c.wav")
        with self.assertRaises(render_mod.RenderError):
            render_mod.listening_copy(source, self.root / "c.wav")

    def test_what_is_not_held_constant_is_recorded(self):
        render_id, directory = render_mod.new_render_dir(self.root)
        record = render_mod.Record(render_id, directory)
        record.not_constant("seed", 1, 2, against="the earlier baseline")
        record.finish()
        stored = json.loads((directory / "render.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["not_held_constant"][0]["field"], "seed")

    def test_a_failure_is_recorded_rather_than_swallowed(self):
        render_id, directory = render_mod.new_render_dir(self.root)
        record = render_mod.Record(render_id, directory)
        record.fail(ValueError("the engine refused"))
        record.finish()
        stored = json.loads((directory / "render.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["failures"][0]["type"], "ValueError")

    def test_a_record_never_claims_more_than_synthetic(self):
        render_id, directory = render_mod.new_render_dir(self.root)
        record = render_mod.Record(render_id, directory)
        record.finish()
        stored = json.loads((directory / "render.json").read_text(encoding="utf-8"))
        self.assertIn("Not an authentic performance", stored["label"])
        self.assertEqual(stored["listening_status"], render_mod.NOT_HEARD)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.audio = self.root / "render.wav"
        sf.write(str(self.audio), np.zeros(4410) + 0.1, 44100, subtype="PCM_24")
        self.ledger = feedback.Ledger(self.root / "feedback.json")

    def test_a_render_with_no_feedback_is_not_heard_not_acceptable(self):
        status = self.ledger.status("r1", self.audio)
        self.assertFalse(status["heard"])
        self.assertIsNone(status["disposition"])
        self.assertIn("not the same as acceptable", status["note"])

    def test_feedback_binds_to_the_bytes_that_were_heard(self):
        self.ledger.add(render_id="r1", audio_path=self.audio, author="Krish",
                        text="the words are clear", disposition=feedback.PREFERRED)
        self.assertTrue(self.ledger.status("r1", self.audio)["heard"])
        sf.write(str(self.audio), np.zeros(4410) + 0.2, 44100, subtype="PCM_24")
        stale = self.ledger.status("r1", self.audio)
        self.assertIn("stale_entries", stale)
        self.assertIn("no longer exists", stale["note"])

    def test_no_preference_is_an_answer(self):
        entry = self.ledger.add(render_id="r1", audio_path=self.audio, author="Krish",
                                text="cannot tell them apart",
                                disposition=feedback.NO_PREFERENCE)
        self.assertEqual(entry["disposition"], feedback.NO_PREFERENCE)
        self.assertTrue(self.ledger.status("r1", self.audio)["heard"])

    def test_feedback_needs_an_author_and_something_said(self):
        for fields in ({"author": "", "text": "x"},
                       {"author": "Krish", "text": "   "}):
            with self.subTest(fields=fields), self.assertRaises(feedback.FeedbackError):
                self.ledger.add(render_id="r1", audio_path=self.audio,
                                disposition=feedback.PREFERRED, **fields)

    def test_an_unknown_disposition_is_refused(self):
        with self.assertRaises(feedback.FeedbackError):
            self.ledger.add(render_id="r1", audio_path=self.audio, author="Krish",
                            text="fine", disposition="pretty good")

    def test_the_four_aspects_are_kept_apart(self):
        entry = self.ledger.add(
            render_id="r1", audio_path=self.audio, author="Krish", text="mixed",
            disposition=feedback.NEEDS_WORK,
            aspects={"words": "clear", "voice_resemblance": "not close",
                     "artifacts_or_ringing": "some ringing on the long note"})
        self.assertEqual(entry["aspects"]["words"], "clear")
        with self.assertRaises(feedback.FeedbackError):
            self.ledger.add(render_id="r1", audio_path=self.audio, author="K",
                            text="x", disposition=feedback.PREFERRED,
                            aspects={"overall": "good"})

    def test_feedback_says_plainly_that_it_is_not_a_weight(self):
        entry = self.ledger.add(render_id="r1", audio_path=self.audio, author="Krish",
                                text="better", disposition=feedback.PREFERRED)
        self.assertIn("updates no model weights", entry["note"])

    def test_the_ledger_round_trips(self):
        self.ledger.add(render_id="r1", audio_path=self.audio, author="Krish",
                        text="good", disposition=feedback.PREFERRED)
        path = self.ledger.save()
        reopened = feedback.Ledger(path)
        self.assertEqual(len(reopened.entries), 1)
        self.assertEqual(reopened.status("r1", self.audio)["disposition"],
                         feedback.PREFERRED)


@unittest.skipUnless(BANK_ZIP.is_file() and DLL.is_file(),
                     "the guide bank or engine is not present")
class CarrierTests(unittest.TestCase):
    """Real synthesis, no model and no GPU. The rest check is the point."""

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.frontend = pipeline.load_tamil_frontend()
        self.bank = pipeline.Bank(BANK_ZIP, DLL)

    def slots(self):
        return [{"draft_articulation": "நி", "word_initial": True},
                {"draft_articulation": "லா", "word_initial": False}]

    def test_a_rest_is_silent_in_the_rendered_audio(self):
        """The reviewed implementation sang through rests at RMS 0.311."""
        melody = tune_mod.from_events([Event(0.0, 0.8, midi=64),
                                       Event(0.8, 0.6, midi=None),
                                       Event(1.4, 0.8, midi=67)])
        slots = self.slots()
        requests = self.frontend.slot_requests(slots, self.bank.aliases())
        parsed = [self.frontend.parse(s["draft_articulation"])[0] for s in slots]
        placed = align.place(parsed, melody)
        result = carrier.render(placed, melody, requests, bank_zip=BANK_ZIP,
                                dll_path=DLL, destination=self.root / "guide.wav")
        self.assertEqual(result["rest_check"]["rest_peak"], 0.0,
                         "a rest must be silent in the audio, not only in the record")
        self.assertGreater(result["rest_check"]["rest_hops"], 0)

    def test_the_guide_leaves_headroom(self):
        melody = tune_mod.from_events([Event(0.0, 0.8, midi=64),
                                       Event(0.8, 0.8, midi=67)])
        slots = self.slots()
        requests = self.frontend.slot_requests(slots, self.bank.aliases())
        parsed = [self.frontend.parse(s["draft_articulation"])[0] for s in slots]
        placed = align.place(parsed, melody)
        result = carrier.render(placed, melody, requests, bank_zip=BANK_ZIP,
                                dll_path=DLL, destination=self.root / "g.wav")
        self.assertEqual(result["samples_at_rail"], 0,
                         "a render that sits on the rail cannot be fixed later")
        self.assertLessEqual(result["peak"], carrier.PEAK_TARGET + 1e-6)


# Imported late so the skip decorator can run without it.
from voice_prototype import carrier  # noqa: E402

if __name__ == "__main__":
    unittest.main()
