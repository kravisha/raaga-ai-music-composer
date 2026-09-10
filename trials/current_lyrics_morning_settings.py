"""Step 2 — current lyrics and tune through the recovered morning settings.

The frozen C conversion input, unchanged, through the setting package the
morning control just proved reproduces exactly. **A synthetic current-lyrics
experiment with recovered morning settings — not a recreation of the morning
human delivery.**

The guide is used **exactly as it is**: not regenerated, not halved again, not
transposed, no D tail envelope, no note events altered, and not replaced by the
morning human source. It already carries the current words, the 12.5-second
melody with its rest, and C's global ×0.5.

**What actually changed against C, all of it, stated together:**

| | C | here |
| --- | --- | --- |
| CFG | 0.7 | **0.3** |
| Compute / interop threads | 6 / 2 | **4 / 1** |
| Raw save | PCM24 | **FLOAT** |

This is a package of three wrapper differences, **not one numerical knob**, and
lower guidance may affect identity or clarity in either direction. FLOAT saving
preserves unclamped model output, so an excursion beyond ±1 in the raw is not
the same thing as a clipped PCM sample — the two are counted separately below.
Input, model, reference, tune and timing are fixed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

QUALITY = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Vocal-Quality-20260909")
ROOT = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Personal-Voice-Trial-20260908")
WORKSPACE = Path(__file__).resolve().parent.parent

C_RENDER = WORKSPACE / "renders" / "r20260910T003437Z-01"
GUIDE = C_RENDER / "guide-input-0.5.wav"
C_LISTENING = C_RENDER / "listening" / "C-converted-headroom-minus18.wav"
REFERENCE = QUALITY / "artist-reference-20260909" / "spb-credited-reference-8s.wav"
CHECKPOINT = ROOT / "models" / "Plachta--Seed-VC" / \
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
CONFIG = ROOT / "inputs" / "local-config.yml"

EXPECTED = {
    GUIDE: "f5654b7ab5a0beb76a6c8ba6fa1f17f2f2825c170ce4b87e8b1dc01b8a8ca15b",
    REFERENCE: "7ba960f2cf3dcbd39166c1209d762e4b5ba9d6f07f37adee22e0c645d5161920",
    CHECKPOINT: "42aef93ffe65857c840d270252fa040f7ba04514945ec460f3ac1ac2a96de684",
    CONFIG: "fbca8ccaea2554894769777b822029d8a4e9290cab0bac1b10d9eac180a6e1cb",
    C_LISTENING: "bcb5f6a5d44281bd3a20bd585a93b15b33eee369b55e16b7df0d2c55aed9e046",
}

SEED = 20260909
SR = 44100
LISTEN_TARGET_DBFS = -18.0
PEAK_CAP = 0.99

OUT = WORKSPACE / "renders" / "current-lyrics-morning-settings"
RAW = OUT / "raw"
STATUS = OUT / "run-status.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; this output is written once.")
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        if sha(path) != expected:
            raise SystemExit(f"{path.name} is not the expected bytes.")

    RAW.mkdir(parents=True)
    report = {
        "status": "loading",
        "what": ("Current lyrics and tune through the recovered morning settings. "
                 "A synthetic experiment, NOT a recreation of the morning human "
                 "delivery."),
        "words": "நிலவாய் நீ வந்தால் / இரவும் பகலாகும்",
        "guide": str(GUIDE),
        "guide_note": ("Used exactly as frozen for C: not regenerated, not halved "
                       "again, not transposed, no tail envelope, notes unchanged, "
                       "and not replaced by the morning human source."),
        "parent_render": "r20260910T003437Z-01",
        "runner": str(Path(__file__).resolve()),
        "pid": os.getpid(),
        "command": " ".join([sys.executable, "-B", "-m",
                             "trials.current_lyrics_morning_settings"]),
        "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hashes": {str(p): h for p, h in EXPECTED.items()},
        "settings": {"inference_cfg_rate": 0.3, "diffusion_steps": 30,
                     "length_adjust": 1.0, "f0_condition": True,
                     "auto_f0_adjust": False, "semi_tone_shift": 0, "fp16": False,
                     "seed": SEED, "device": "cpu", "precision": "float32",
                     "torch_threads": 4, "torch_interop_threads": 1,
                     "raw_save_subtype": "FLOAT"},
        "changed_against_C": {
            "inference_cfg_rate": {"C": 0.7, "here": 0.3},
            "torch_threads": {"C": 6, "here": 4},
            "torch_interop_threads": {"C": 2, "here": 1},
            "raw_save_subtype": {"C": "PCM_24", "here": "FLOAT"},
            "note": ("Three wrapper differences together, not one numerical knob. "
                     "Audible differences must not be attributed to CFG alone."),
        },
        "unchanged_against_C": ("guide bytes, reference, checkpoint, config, "
                                "words, melody, timing, rest, global 0.5 headroom"),
        "network_used": False, "training": False,
        "installed_code_modified": False,
    }

    def update(**fields):
        report.update(fields)
        STATUS.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(json.dumps(fields, default=str, ensure_ascii=False), flush=True)

    update()
    started = time.perf_counter()
    try:
        os.chdir(ROOT)
        os.environ.update(
            CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
            HF_HOME=str(ROOT / "cache/hf"), TORCH_HOME=str(ROOT / "cache/torch"),
            NUMBA_CACHE_DIR=str(OUT / "cache/numba"),
            MPLCONFIGDIR=str(OUT / "cache/matplotlib"))
        sys.path.insert(0, str(ROOT / "runtime-deps"))
        sys.path.insert(0, str(ROOT / "runtime-source"))

        import numpy as np
        import soundfile as sf
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        import inference
        inference.device = torch.device("cpu")

        def local_model(repo_id, model_filename="pytorch_model.bin",
                        config_filename=None):
            directory = ROOT / "models" / repo_id.replace("/", "--")
            path = directory / model_filename
            assert path.is_file()
            if config_filename is None:
                return str(path)
            configuration = directory / config_filename
            assert configuration.is_file()
            return str(path), str(configuration)

        inference.load_custom_model_from_hf = local_model
        written: list = []

        def save_audio(path, tensor, sample_rate, **kw):
            """FLOAT, so the model's own output is kept unclamped."""
            path = Path(path).resolve()
            assert path.is_relative_to(RAW.resolve()) and not path.exists()
            values = tensor.detach().cpu().numpy().T
            assert np.isfinite(values).all()
            sf.write(str(path), values, sample_rate, subtype="FLOAT")
            written.append(path)

        inference.torchaudio.save = save_audio

        args = SimpleNamespace(
            source=str(GUIDE), target=str(REFERENCE), output=str(RAW),
            diffusion_steps=30, length_adjust=1., inference_cfg_rate=.3,
            f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
            checkpoint=str(CHECKPOINT), config=str(CONFIG), fp16=False)
        update(status="converting")
        inference.main(args)
        if len(written) != 1:
            raise RuntimeError(f"expected one output, got {len(written)}")
        raw_path = written[0]

        raw, rate = sf.read(str(raw_path), dtype="float64")
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        raw_peak = float(np.max(np.abs(raw)))
        raw_measure = {
            "path": str(raw_path), "sha256": sha(raw_path), "subtype": "FLOAT",
            "frames": int(len(raw)), "sample_rate": int(rate),
            "seconds": round(len(raw) / rate, 6),
            "finite": bool(np.isfinite(raw).all()),
            "peak": raw_peak,
            "rms_dbfs": float(20 * np.log10(float(np.sqrt(np.mean(raw ** 2))))),
            "samples_beyond_full_scale": int(np.sum(np.abs(raw) > 1.0)),
            "excursion_note": ("FLOAT keeps the model's unclamped output. A value "
                               "beyond 1.0 here is an excursion, NOT an already "
                               "clipped PCM sample. Nothing has been clamped."),
        }

        # -- listening copy at -18 dBFS RMS, scalar gain only
        rms = float(np.sqrt(np.mean(raw ** 2)))
        gain = 10 ** (LISTEN_TARGET_DBFS / 20) / rms
        scaled = raw * gain
        peak_after = float(np.max(np.abs(scaled)))
        capped = peak_after > PEAK_CAP
        applied_target = LISTEN_TARGET_DBFS
        if capped:
            # Act, do not merely announce. An earlier version set this flag and
            # then wrote the unadjusted samples anyway, which would have reported
            # a cap that never happened. The whole comparison drops to a common
            # lower level instead - scalar only, both new copies, and preserved C
            # is never rewritten.
            applied_target = LISTEN_TARGET_DBFS - float(
                20 * np.log10(peak_after / PEAK_CAP))
            gain = 10 ** (applied_target / 20) / rms
            scaled = raw * gain
            peak_after = float(np.max(np.abs(scaled)))
        listening = OUT / "E-current-lyrics-cfg03-minus18.wav"
        sf.write(str(listening), scaled, SR, subtype="PCM_24")
        final, _ = sf.read(str(listening), dtype="float64")
        if final.ndim > 1:
            final = final.mean(axis=1)
        rail_p, rail_n = 1.0 - 2.0 ** -23, -1.0
        listening_measure = {
            "path": str(listening), "sha256": sha(listening), "subtype": "PCM_24",
            "target_rms_dbfs": LISTEN_TARGET_DBFS,
            "applied_target_rms_dbfs": applied_target,
            "gain_db": float(20 * np.log10(gain)),
            "processing": "one scalar gain only; no filter, EQ, limiter or new target",
            "peak_after_gain_before_write": peak_after,
            "peak_cap_exceeded": capped,
            "peak_cap": PEAK_CAP,
            "achieved_rms_dbfs": float(20 * np.log10(float(np.sqrt(np.mean(final ** 2))))),
            "achieved_peak": float(np.max(np.abs(final))),
            "pcm24_samples_at_positive_rail": int(np.sum(final >= rail_p)),
            "pcm24_samples_at_negative_rail": int(np.sum(final <= rail_n)),
            "compare_with": str(C_LISTENING),
            "compare_with_sha256": EXPECTED[C_LISTENING],
            "c_rails_note": ("C's preserved raw already holds 117 PCM24 rail "
                             "samples; that is irreversible in its input to "
                             "normalisation and no later gain undoes it."),
        }

        for path, expected in EXPECTED.items():
            assert sha(path) == expected, f"{path} changed during the run"

        update(status="completed", exit_code=0,
               elapsed_seconds=time.perf_counter() - started,
               finish_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               raw=raw_measure, listening=listening_measure,
               inputs_unchanged=True,
               label=("Synthetic current-lyrics experiment with recovered morning "
                      "settings. Not a recreation of the morning human delivery, "
                      "not an acceptance, and not a claim about identity."))
    except BaseException as exc:                    # noqa: BLE001
        update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               exit_code=1, error=repr(exc),
               elapsed_seconds=time.perf_counter() - started)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
