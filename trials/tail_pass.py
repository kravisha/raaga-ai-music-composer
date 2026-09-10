"""One local tail attenuation pass — candidate D.

Krish, superseding the Ira-only direction: *the first eight seconds are better,
the last four seconds are completely bad, and it starts getting worse
gradually.* C still preferred to B, not accepted.

**The one control:** a smooth gain envelope of **-4.0 dB over 8.0-12.5 s**,
raised-cosine 30 ms edges, applied to the frozen C guide bytes, then C's same
global x0.5. The first 8.0 s are bit-identical to what C was fed - verified
against C's actual input file, maximum difference 0.000e+00. Nothing is
re-synthesised, so pitch and timing are untouched by construction and the
segment Krish called better is the same bytes rather than merely similar.

Direction chosen from evidence: B to C showed less input gives fewer rails
(174 to 117), so the tail is attenuated rather than raised.

**What this does not claim.** Level differences between the head and the tail do
not establish a cause for what Krish hears, and phrases 0-3 s and 8-12 s are
different material rather than the same material treated differently. This tests
one amplitude hypothesis and nothing more. It is a local amplitude control, not
evidence of learned voicing, and no weights are touched.

The `syllable_volume` control in `carrier.py` and `trials/ira_presence_pass.py`
are both preserved and deliberately unused here: the first re-synthesises, and
re-synthesis measurably changed phrase 0; the second was superseded before any
model run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_prototype import render as render_mod                # noqa: E402
from voice_prototype.convert import CHECKPOINT, CONFIG, SETTINGS, SeedVC  # noqa: E402
from trials.headroom_pass import PassError, measure_pcm         # noqa: E402
from trials.nilavaai_two_lines import REFERENCE                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE_RENDER = ROOT / "renders" / "r20260910T001153Z-01"      # holds the guide
C_RENDER = ROOT / "renders" / "r20260910T003437Z-01"         # the parent, C
C_ID = "r20260910T003437Z-01"

GUIDE = BASE_RENDER / "guide.wav"
GUIDE_SHA = "4d66e47565759364e09a3958dcdbbaba279862785fbab29c90394aec3a36df91"
C_LISTENING = C_RENDER / "listening" / "C-converted-headroom-minus18.wav"
C_LISTENING_SHA = "bcb5f6a5d44281bd3a20bd585a93b15b33eee369b55e16b7df0d2c55aed9e046"

#: The Ira interval, from the alignment - இ starts at 6.30, ர ends at 7.50.
IRA_START_S = 8.00
IRA_END_S = 12.50
#: Modest, and chosen from the measurement above rather than by taste: +4 dB
#: lifts ர from -19.12 to about -15.1 and இ from -16.48 to about -12.5, which
#: brings them toward the phrase mean of -12.17 without passing வும் at -6.28.
IRA_GAIN_DB = -4.0
#: Raised-cosine edges, so the envelope introduces no step and no click.
RAMP_S = 0.030
#: C's global input gain, unchanged.
GLOBAL_GAIN = 0.5
C_INPUT_PEAK = 0.4450000525
SEED = 20260909


def envelope(length: int, rate: int) -> np.ndarray:
    """1.0 everywhere except a smoothly raised window over the Ira interval."""
    gain = 10 ** (IRA_GAIN_DB / 20.0)
    shape = np.ones(length)
    begin, end = int(IRA_START_S * rate), int(IRA_END_S * rate)
    ramp = max(1, int(RAMP_S * rate))
    shape[begin:end] = gain
    rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
    shape[begin:begin + ramp] = 1 + (gain - 1) * rise
    shape[end - ramp:end] = 1 + (gain - 1) * rise[::-1]
    return shape


def build_input(destination: Path) -> dict:
    """The frozen guide, one local envelope, then C's global half."""
    audio, rate = sf.read(str(GUIDE), dtype="float64")
    if audio.ndim > 1:
        raise PassError("The guide is expected to be mono.")
    shaped = audio * envelope(len(audio), rate) * GLOBAL_GAIN
    peak = float(np.max(np.abs(shaped)))
    if peak > C_INPUT_PEAK + 1e-9:
        raise PassError(
            f"The shaped input peaks at {peak:.7f}, above C's {C_INPUT_PEAK:.7f}. "
            f"This pass may not raise the level the conversion is handed.")
    sf.write(str(destination), shaped, rate, subtype="PCM_24")

    # Everything outside the interval must be C's input exactly.
    baseline_half = audio * GLOBAL_GAIN
    written, _ = sf.read(str(destination), dtype="float64")
    outside = np.ones(len(audio), dtype=bool)
    outside[int(IRA_START_S * rate):int(IRA_END_S * rate)] = False
    quantum = 2.0 ** -23
    drift = float(np.max(np.abs(written[outside] - baseline_half[outside])))
    if drift > quantum:
        raise PassError(
            f"Samples outside the Ira interval moved by {drift:.3e}, more than one "
            f"24-bit step. Only the named interval may change.")
    return {"rate": rate, "frames": len(audio), "peak": peak,
            "outside_max_drift": drift, "one_pcm24_step": quantum}


def main(convert_factory=None, out_root: Path | None = None) -> int:
    out_root = Path(out_root) if out_root is not None else ROOT / "renders"
    if render_mod.sha256(GUIDE) != GUIDE_SHA:
        raise PassError("The baseline guide is not the expected bytes.")
    if render_mod.sha256(C_LISTENING) != C_LISTENING_SHA:
        raise PassError("C's listening copy is not the bytes Krish heard.")
    c_record = json.loads((C_RENDER / "render.json").read_text(encoding="utf-8"))
    for name, expected in SETTINGS.items():
        if c_record["settings"].get(name) != expected:
            raise PassError(f"C ran with a different {name}; settings must match C.")
    if c_record["settings"].get("seed") != SEED:
        raise PassError("C's seed is not the one this pass would use.")
    for name, path in (("reference", Path(REFERENCE.path)),
                       ("checkpoint", CHECKPOINT), ("conversion_config", CONFIG)):
        recorded = (c_record["inputs"].get(name) or {}).get("sha256")
        if recorded != render_mod.sha256(path):
            raise PassError(f"{name} differs from the bytes C used.")

    render_id, directory = render_mod.new_render_dir(out_root, parent=C_ID)
    record = render_mod.Record(render_id, directory, parent=C_ID)
    record.data["contract"] = "docs/spec/VOICE_PROTOTYPE_CONTRACT.md at 46e5ca2 (v2)"
    record.data["requested_by"] = "Krish, relayed and coordinated by Arya"
    record.data["pass"] = "Ira presence: one local gain envelope, one conversion"
    record.data["control_change"] = {
        "field": "local gain envelope on the frozen guide, tail only",
        "interval_s": [IRA_START_S, IRA_END_S],
        "gain_db": IRA_GAIN_DB,
        "ramp_s": RAMP_S,
        "ramp_shape": "raised cosine, so no step and no click",
        "before": "0 dB across the whole guide",
        "after": f"+{IRA_GAIN_DB} dB over {IRA_START_S}-{IRA_END_S} s only",
        "chosen_because": (
            "Krish: the first eight seconds are better, the last four are "
            "completely bad and it gets worse gradually. Measured in C: the tail "
            "output runs -5.5 to -8.9 dBFS against -8.2 to -11.2 in the good head, "
            "peaks at exactly 1.0, and holds 112 of the 117 clipped samples "
            "between 7.08 and 10.73 s - while the guide is at its QUIETEST there. "
            "The conversion drives the tail hardest where it is fed least. B to C "
            "showed less input gives fewer rails (174 to 117), so this attenuates "
            "the tail rather than raising it."),
        "hypothesis_not_fact": (
            "That the tail distortion follows the drive into that region. If the "
            "conversion simply renormalises the quieter tail back up, this will "
            "not help, and that outcome is informative and will be reported as "
            "such."),
        "applied_to": "the frozen baseline guide bytes; nothing is re-synthesised",
        "then": f"C's unchanged global x{GLOBAL_GAIN} input gain",
        "not_this": ("not a pitch change, not a transposition, not a timing change, "
                     "not a re-synthesis, and not a whole-file renormalisation"),
        "not_evidence_of": ("learned voicing. It is a local amplitude control, and "
                            "no weights were touched."),
    }
    record.data["feedback_binding"] = {
        "parent_render": C_ID,
        "krish_heard": str(C_LISTENING), "krish_heard_sha256": C_LISTENING_SHA,
        "verdict": ("Superseding: the first eight seconds are better, the last "
                    "four seconds are completely bad, and it starts getting worse "
                    "gradually. C still preferred to B, not accepted."),
        "acted_on_in_this_pass": "tail level only, 8.0-12.5 s; the first eight seconds Krish called better are byte-identical to C's input",
    }

    try:
        shaped = directory / "guide-input-tail-4dB-half.wav"
        built = build_input(shaped)
        record.input("frozen_guide", path=GUIDE, **measure_pcm(GUIDE))
        record.input("shaped_input", path=shaped, **measure_pcm(shaped))
        record.stage("shaping", **built, processing=(
            "gain envelope over one interval, then the same global 0.5 as C; no "
            "EQ, limiting, resampling, renormalisation or re-synthesis"))
        record.input("reference", **REFERENCE.verify())

        convert = (convert_factory() if convert_factory is not None
                   else SeedVC(seed=SEED, compared_baseline="spb_conditioned"))
        converted = convert(shaped, REFERENCE, directory, record)
        out = measure_pcm(converted)
        record.output("converted", converted, **out)
        if out["samples_at_positive_rail"] or out["samples_at_negative_rail"]:
            record.warn(
                f"Clipped: {out['samples_at_positive_rail']} positive and "
                f"{out['samples_at_negative_rail']} negative samples at the rail. "
                f"Reported, not repaired.")
        copy = render_mod.listening_copy(
            converted, directory / "listening" / "D-tail-minus18.wav")
        record.output("listening_converted", copy["to"], **copy)
        record.data["level_matched_comparison"] = {
            "C": {"path": str(C_LISTENING), "sha256": C_LISTENING_SHA},
            "D": copy,
            "note": "both at -18 dBFS; listening gains are not the experimental gain"}
    except Exception as exc:                        # noqa: BLE001
        record.fail(exc)
        record.finish(status="failed")
        raise

    record.finish()
    print(json.dumps({"render_id": render_id, "parent": C_ID,
                      "input_peak": built["peak"], "c_input_peak": C_INPUT_PEAK,
                      "output_peak": out["peak"],
                      "rails": [out["samples_at_positive_rail"],
                                out["samples_at_negative_rail"]],
                      "listening": copy["to"], "record": str(record.path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
