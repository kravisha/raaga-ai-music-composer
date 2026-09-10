"""One authorized headroom comparison — exactly one change, exactly one run.

The baseline guide is **not regenerated**. Its bytes are read, multiplied by
0.5, and written to a new input; everything else — lyric, melody, timing, words,
reference, checkpoint, config, seed, device, precision, steps, CFG, F0
conditioning — is copied from the baseline render's own record and verified
against it before the run.

The question is narrow and worth asking on its own: the baseline guide peaked at
0.890 and its conversion came back with 174 samples on the rail. Does handing
the conversion a quieter input give it room? **A lower guide does not guarantee
fewer clipped samples**, and if it stays clipped this pass says so and stops.

Nothing here is a sweep, an articulation change, a character change or training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_prototype import render as render_mod                # noqa: E402
from voice_prototype.convert import SETTINGS, SeedVC            # noqa: E402
from voice_prototype.convert import CHECKPOINT, CONFIG          # noqa: E402
from trials.nilavaai_two_lines import REFERENCE                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "renders" / "r20260910T001153Z-01"
BASELINE_ID = "r20260910T001153Z-01"

GUIDE = BASELINE / "guide.wav"
GUIDE_SHA = "4d66e47565759364e09a3958dcdbbaba279862785fbab29c90394aec3a36df91"
RAW_B = BASELINE / "conversion-output" / "vc_guide_spb-credited-reference-8s_1.0_30_0.7.wav"
LISTENING_B = BASELINE / "listening" / "B-converted-minus18.wav"

#: The one change. Linear, on the input, before inference.
GUIDE_GAIN = 0.5
SEED = 20260909

#: The actual extremes of signed 24-bit PCM read back as float in [-1, 1].
#: The positive rail is (2**23 - 1) / 2**23 - one step short of 1.0 - and the
#: negative rail is exactly -1.0. Counting at 0.999999 instead answers a
#: different, looser question.
PCM24_POSITIVE_RAIL = 1.0 - 2.0 ** -23
PCM24_NEGATIVE_RAIL = -1.0
NEAR_RAIL = 0.999999


class PassError(Exception):
    """A refusal. Never a silent correction."""


def measure_pcm(path: Path) -> dict:
    """Peaks and rail counts on both signs, from the file as written.

    Deliberately returns no `path` or `sha256`: the record adds those itself,
    and returning them here made every call site pass `path` twice.

    Rails are counted at the true 24-bit extremes. The looser 0.999999 count is
    kept beside them as a diagnostic rather than in place of them, because the
    two answer different questions and only one of them is "clipped".
    """
    audio, rate = sf.read(str(path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return {
        "frames": int(len(audio)), "sample_rate": int(rate),
        "seconds": round(len(audio) / rate, 6),
        "finite": bool(np.isfinite(audio).all()),
        "peak": round(float(np.max(np.abs(audio))), 10),
        "peak_positive": round(float(np.max(audio)), 10),
        "peak_negative": round(float(np.min(audio)), 10),
        "samples_at_positive_rail": int(np.sum(audio >= PCM24_POSITIVE_RAIL)),
        "samples_at_negative_rail": int(np.sum(audio <= PCM24_NEGATIVE_RAIL)),
        "rail_definition": {"positive": PCM24_POSITIVE_RAIL,
                            "negative": PCM24_NEGATIVE_RAIL,
                            "note": "signed 24-bit PCM extremes, not a 0.999999 proxy"},
        "diagnostic_near_rail_positive": int(np.sum(audio >= NEAR_RAIL)),
        "diagnostic_near_rail_negative": int(np.sum(audio <= -NEAR_RAIL)),
        "rms_dbfs": round(float(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))),
                                                  1e-12))), 4),
    }


def main(convert_factory=None, *, out_root: Path | None = None,
         baseline: Path | None = None) -> int:
    """`convert_factory` exists so the bookkeeping around the run can be
    exercised with a stand-in. The real model is the default."""
    global BASELINE, BASELINE_ID
    if baseline is not None:
        BASELINE = Path(baseline)
        BASELINE_ID = BASELINE.name
    out_root = Path(out_root) if out_root is not None else ROOT / "renders"
    if render_mod.sha256(GUIDE) != GUIDE_SHA:
        raise PassError(
            f"{GUIDE} is not the baseline guide. This comparison is only "
            f"meaningful against the exact bytes that produced B.")
    baseline_record = json.loads((BASELINE / "render.json").read_text(encoding="utf-8"))
    for name, expected in SETTINGS.items():
        actual = baseline_record["settings"].get(name)
        if actual != expected:
            raise PassError(
                f"The baseline ran with {name}={actual!r}, not {expected!r}. "
                f"Copying settings means copying them, not assuming them.")
    if baseline_record["settings"].get("seed") != SEED:
        raise PassError(
            f"The baseline seed is {baseline_record['settings'].get('seed')!r}, "
            f"not {SEED}.")

    for name, path in (("reference", Path(REFERENCE.path)),
                       ("checkpoint", CHECKPOINT), ("conversion_config", CONFIG)):
        recorded = (baseline_record["inputs"].get(name) or {}).get("sha256")
        actual = render_mod.sha256(path) if Path(path).is_file() else None
        if recorded is None:
            raise PassError(f"The baseline record holds no hash for {name}.")
        if actual != recorded:
            raise PassError(
                f"{name} is {str(actual)[:12]} now and was {recorded[:12]} in the "
                f"baseline. This comparison needs the same bytes on both sides.")

    render_id, directory = render_mod.new_render_dir(out_root, parent=BASELINE_ID)
    record = render_mod.Record(render_id, directory, parent=BASELINE_ID)
    record.data["contract"] = "docs/spec/VOICE_PROTOTYPE_CONTRACT.md at 46e5ca2 (v2)"
    record.data["requested_by"] = "Krish, relayed and coordinated by Arya"
    record.data["kind"] = "revised_render"
    record.data["pass"] = "headroom: one control changed, one conversion"
    record.data["notes"] = (
        "The baseline guide was attenuated by 0.5 before inference. It was not "
        "regenerated: the same bytes, scaled. Everything else is copied from the "
        "baseline record and verified before the run.")
    record.data["control_change"] = {
        "field": "guide input gain before inference",
        "before": 1.0, "after": GUIDE_GAIN,
        "decibels": round(float(20 * np.log10(GUIDE_GAIN)), 3),
        "why": ("The baseline conversion returned 174 samples on the rail from a "
                "guide that peaked at 0.890. This asks whether a quieter input "
                "gives the conversion room."),
        "not_this": ("This is attenuation of the INPUT before a fresh inference. "
                     "It is not attenuation of the already-clipped output, which "
                     "would be quieter and still clipped."),
        "unchanged": ("lyric, melody, timing, words, reference bytes, checkpoint, "
                      "config, seed, device, precision, diffusion steps, length "
                      "adjust, CFG rate, F0 conditioning, auto F0 adjust, "
                      "semitone shift"),
    }
    record.data["feedback_binding"] = {
        "parent_render": BASELINE_ID,
        "krish_heard": str(LISTENING_B),
        "krish_heard_sha256": render_mod.sha256(LISTENING_B),
        "verdict": ("A is bad, B is better, both unacceptable; B is a relative "
                    "baseline only. Articulation, character and a mechanical 'na' "
                    "remain unsatisfactory. Tune acceptable for now."),
        "acted_on_in_this_pass": "input level only",
        "retained_for_later": ("articulation, voice character, the mechanical "
                               "'na' - not acted on here"),
    }

    try:
        audio, rate = sf.read(str(GUIDE), dtype="float64")
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        scaled = audio * GUIDE_GAIN
        attenuated = directory / "guide-input-0.5.wav"
        sf.write(str(attenuated), scaled, rate, subtype="PCM_24")

        before, after = measure_pcm(GUIDE), measure_pcm(attenuated)
        if before["frames"] != after["frames"] or before["sample_rate"] != after["sample_rate"]:
            raise PassError("The attenuated input changed length or rate.")
        ratio = after["peak"] / before["peak"]
        record.input("baseline_guide", path=GUIDE, **before)
        record.input("attenuated_guide", path=attenuated, channels=channels,
                     **after)
        # `before` and `after` no longer carry `path`; the record supplies it.
        record.stage("attenuation", gain_linear=GUIDE_GAIN,
                     gain_db=round(float(20 * np.log10(GUIDE_GAIN)), 3),
                     measured_peak_ratio=round(float(ratio), 6),
                     expected_peak_ratio=GUIDE_GAIN,
                     processing="gain only; no EQ, limiting, resampling or re-render",
                     note=("PCM_24 quantisation is the only other difference; the "
                           "waveform shape, rests, frame count, rate and channel "
                           "count are unchanged."))
        record.input("reference", **REFERENCE.verify())

        convert = (convert_factory() if convert_factory is not None
                   else SeedVC(seed=SEED, compared_baseline="spb_conditioned"))
        converted = convert(attenuated, REFERENCE, directory, record)

        out = measure_pcm(converted)
        record.output("converted", converted, **out)
        if out["samples_at_positive_rail"] or out["samples_at_negative_rail"]:
            record.warn(
                f"Still clipped: {out['samples_at_positive_rail']} positive and "
                f"{out['samples_at_negative_rail']} negative samples at the rail "
                f"from a guide peaking at {after['peak']:.4f}. A lower input does "
                f"not guarantee fewer clipped samples, and this pass stops here "
                f"rather than starting a sweep.")

        copy = render_mod.listening_copy(
            converted, directory / "listening" / "C-converted-headroom-minus18.wav")
        record.output("listening_converted", copy["to"], **copy)
        baseline_copy = {"path": str(LISTENING_B),
                         "sha256": render_mod.sha256(LISTENING_B),
                         "note": "the baseline B at the same -18 dBFS target, unchanged"}
        record.data["level_matched_comparison"] = {
            "baseline_B": baseline_copy, "new_C": copy,
            "listening_gains_are_separate_from": "the experimental 0.5 guide gain"}
    except Exception as exc:                        # noqa: BLE001 - recorded
        record.fail(exc)
        record.finish(status="failed")
        raise

    record.finish()
    print(json.dumps({
        "render_id": render_id, "parent": BASELINE_ID,
        "input_peak": after["peak"], "input_gain": GUIDE_GAIN,
        "output_peak": out["peak"],
        "output_rail_positive": out["samples_at_positive_rail"],
        "output_rail_negative": out["samples_at_negative_rail"],
        "listening": copy["to"], "record": str(record.path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
