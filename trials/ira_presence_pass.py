"""One local presence pass on Ira — candidate D.

Krish: C is better than B; the first Nila phrase is better; the initial **Ira**
of *Iravum Pagal Aagum* is meek and needs strength. "Pushed up" means presence,
**not pitch** — no transposition, no tune change.

Measured before choosing anything, on the original unattenuated guide:

| syllable | guide RMS |
| --- | --- |
| இ  6.30–7.00 s | −16.48 dBFS |
| ர  7.00–7.50 s | **−19.12 dBFS** — the weakest syllable in the whole guide |
| வும் 7.50–8.50 s | −6.28 dBFS |
| phrase 1 mean | −12.17 dBFS |

`ra` sits 12.8 dB under the syllable that follows it. The weakness is in the
guide, before any conversion.

**The one control:** a smooth local gain envelope over 6.30–7.50 s applied
directly to the frozen baseline guide, then C's same global ×0.5. Nothing is
re-synthesised, so the successful first phrase and every other syllable keep
their exact samples; there is no whole-file renormalisation to leak the change
outward. Pitch and timing are untouched by construction — a gain envelope moves
no sample in time and changes no frequency.

The request-volume control in `carrier.py` is deliberately **not used here**: it
re-synthesises, and re-synthesis changed phrase 0. It stays available and unused.

A gain envelope is a local amplitude change. It is not evidence of learned
voicing, and nothing here trains anything.
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
IRA_START_S = 6.30
IRA_END_S = 7.50
#: Modest, and chosen from the measurement above rather than by taste: +4 dB
#: lifts ர from -19.12 to about -15.1 and இ from -16.48 to about -12.5, which
#: brings them toward the phrase mean of -12.17 without passing வும் at -6.28.
IRA_GAIN_DB = 4.0
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
        "field": "local gain envelope on the frozen guide, Ira only",
        "interval_s": [IRA_START_S, IRA_END_S],
        "gain_db": IRA_GAIN_DB,
        "ramp_s": RAMP_S,
        "ramp_shape": "raised cosine, so no step and no click",
        "before": "0 dB across the whole guide",
        "after": f"+{IRA_GAIN_DB} dB over {IRA_START_S}-{IRA_END_S} s only",
        "chosen_because": ("measured: ra -19.12 dBFS is the weakest syllable in "
                           "the guide and sits 12.8 dB under the vum that follows "
                           "it; phrase 1 averages -12.17 dBFS"),
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
        "verdict": ("C is definitely better than B, said twice; continue. First "
                    "Nila phrase better. Initial Ira of the second phrase is "
                    "meek/weak and needs strength. Preferred comparison, not "
                    "final acceptance."),
        "acted_on_in_this_pass": "Ira presence only, 6.30-7.50 s",
    }

    try:
        shaped = directory / "guide-input-ira+4dB-half.wav"
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
            converted, directory / "listening" / "D-ira-presence-minus18.wav")
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
