"""Reproduce the morning preferred-04 conversion — one control run.

A faithful copy of `Raaga-Vocal-Quality-20260909/probe_spb_conversion_core.py`
with `--reduced-guidance`, changed **only** in where it reads its roots from and
where it writes. The historical script is not run in place: its `OUT`/`RAW`/
`STATUS` point at preserved outputs and it refuses a completed status.

Kept faithful, deliberately and item by item: the inference call and its
arguments, CFG **0.3**, 30 steps, the three seeds at 20260909, `torch` 4 threads
with 1 interop, CPU float32, the offline environment, the local-model resolver,
the F0 and vocoder capture wrappers, and the raw save subtype **FLOAT**.

`HERE` is **not** allowed to default to this file's parent. In a copy that would
silently repoint the historical inputs at this workspace, so the quality and
runtime roots are bound explicitly and every input is hash-checked before use.

One conversion. No transposition, no synthetic guide, no second render, no
training, no writes into the historical directory.
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

#: Explicitly bound. Not derived from this file's location.
QUALITY = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Vocal-Quality-20260909")
ROOT = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Personal-Voice-Trial-20260908")
WORKSPACE = Path(__file__).resolve().parent.parent

SOURCE = QUALITY / "listening" / "05-recorded-voice-word-timing-lower-register.wav"
REFERENCE = QUALITY / "artist-reference-20260909" / "spb-credited-reference-8s.wav"
CHECKPOINT = ROOT / "models" / "Plachta--Seed-VC" / \
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
CONFIG = ROOT / "inputs" / "local-config.yml"

#: What the preserved morning artifacts are, checked before and after.
EXPECTED = {
    SOURCE: "e650a6584682328b82a375a2d0a9d02f37510ebff5e236d0f0f9243e7185fa0e",
    REFERENCE: "7ba960f2cf3dcbd39166c1209d762e4b5ba9d6f07f37adee22e0c645d5161920",
    CHECKPOINT: "42aef93ffe65857c840d270252fa040f7ba04514945ec460f3ac1ac2a96de684",
    CONFIG: "fbca8ccaea2554894769777b822029d8a4e9290cab0bac1b10d9eac180a6e1cb",
}

PRESERVED_04_RAW = (QUALITY / "spb-core-resonance-20260909" / "reduced-guidance" /
                    "raw" / "vc_05-recorded-voice-word-timing-lower-register_"
                            "spb-credited-reference-8s_1.0_30_0.3.wav")
PRESERVED_04_RAW_SHA = "c1669e72a947ddcdfc0de815c140456d66909b073c83d5a97b81b42711da74fd"
PRESERVED_04_LISTENING = (QUALITY / "spb-core-resonance-20260909" / "audio" /
                          "04-reduced-core-guidance.wav")
PRESERVED_04_LISTENING_SHA = \
    "1e9da45d7daf5274fbaec7de0bb9b9ed326f021a1593f905eef468c68d3c5f97"

#: The gain the historical listening step recorded, reused rather than
#: recomputed. Labelled wherever it appears: this is the preserved scalar, not a
#: fresh loudness match.
PRESERVED_04_GAIN_DB = -0.779490993561403
SEED = 20260909
SR = 44100

OUT = WORKSPACE / "renders" / "morning-control"
RAW = OUT / "raw"
STATUS = OUT / "control-status.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; this output is written once.")
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        actual = sha(path)
        if actual != expected:
            raise SystemExit(
                f"{path.name} is {actual[:12]}, expected {expected[:12]}. This "
                f"control is only meaningful against the exact morning bytes.")
    for path, expected in ((PRESERVED_04_RAW, PRESERVED_04_RAW_SHA),
                           (PRESERVED_04_LISTENING, PRESERVED_04_LISTENING_SHA)):
        if sha(path) != expected:
            raise SystemExit(f"{path.name} is not the preserved artifact.")

    RAW.mkdir(parents=True)
    report = {
        "status": "loading",
        "what": ("One reproduction of the morning preferred-04 conversion, from "
                 "the preserved human-recorded source. Control first."),
        "runner": str(Path(__file__).resolve()),
        "copied_from": str(QUALITY / "probe_spb_conversion_core.py"),
        "changed_from_historical": ("output and status paths only, plus explicit "
                                    "QUALITY/ROOT binding so a copy cannot "
                                    "repoint the historical inputs"),
        "pid": os.getpid(),
        "command": " ".join([sys.executable, "-B", "-m", "trials.morning_control"]),
        "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(SOURCE), "reference": str(REFERENCE),
        "source_hashes": {str(p): h for p, h in EXPECTED.items()},
        "seed": SEED, "inference_cfg_rate": 0.3,
        "diffusion_steps": 30, "length_adjust": 1.0, "f0_condition": True,
        "auto_f0_adjust": False, "semi_tone_shift": 0, "fp16": False,
        "device": "cpu", "precision": "float32",
        "torch_threads": 4, "torch_interop_threads": 1,
        "raw_save_subtype": "FLOAT",
        "network_used": False, "training": False,
        "installed_code_modified": False,
        "not_this": ("not a transposition, not a synthetic guide, not the current "
                     "C guide, not a second render, and not training"),
    }

    def update(**fields):
        report.update(fields)
        STATUS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fields, default=str), flush=True)

    update()
    started = time.perf_counter()
    try:
        os.chdir(ROOT)
        os.environ.update(
            CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
            HF_HOME=str(ROOT / "cache/hf"), TORCH_HOME=str(ROOT / "cache/torch"),
            # Mutable caches into this workspace, not the historical tree.
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

        capture, loaded = {}, {}
        original_load = inference.load_models

        def load_models(args):
            values = list(original_load(args))
            loaded["vocoder"] = values[3]
            loaded["mel_fn"] = values[5]
            loaded["mel_args"] = values[6]
            old_f0 = values[2]

            def track_f0(*a, **kw):
                result = old_f0(*a, **kw)
                capture["reference_f0" if "reference_f0" not in capture
                        else "source_f0"] = np.array(result)
                return result

            values[2] = track_f0

            def capture_vocoder(mel):
                capture["generated_mel"] = mel.detach().cpu().numpy()
                return loaded["vocoder"](mel)

            values[3] = capture_vocoder
            return tuple(values)

        inference.load_models = load_models

        def save_audio(path, tensor, sample_rate, **kw):
            path = Path(path).resolve()
            assert path.is_relative_to(RAW.resolve()) and not path.exists()
            values = tensor.detach().cpu().numpy().T
            assert np.isfinite(values).all() and abs(values).max() < 10
            sf.write(path, values, sample_rate, subtype="FLOAT")

        inference.torchaudio.save = save_audio

        args = SimpleNamespace(
            source=str(SOURCE), target=str(REFERENCE), output=str(RAW),
            diffusion_steps=30, length_adjust=1., inference_cfg_rate=.3,
            f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
            checkpoint=str(CHECKPOINT), config=str(CONFIG), fp16=False)
        update(status="reproducing_morning_04_with_intermediate_capture")
        inference.main(args)

        update(status="saving_comparable_source_mels")
        with torch.no_grad():
            for name, path in (("source", SOURCE), ("reference", REFERENCE)):
                wave = inference.librosa.load(path, sr=SR)[0]
                mel = loaded["mel_fn"](torch.from_numpy(wave).unsqueeze(0).float())
                capture[name + "_mel"] = mel.cpu().numpy()
        np.savez_compressed(OUT / "core-intermediates.npz", **capture)

        # -- the comparison that matters: against preserved 04, not against 0.7
        new_raw = next(RAW.glob("vc_*.wav"))
        a, sra = sf.read(str(PRESERVED_04_RAW), dtype="float64")
        b, srb = sf.read(str(new_raw), dtype="float64")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if b.ndim > 1:
            b = b.mean(axis=1)
        same_shape = (sra == srb and len(a) == len(b))
        comparison = {
            "against": str(PRESERVED_04_RAW),
            "against_sha256": PRESERVED_04_RAW_SHA,
            "note": ("The historical script's final metric compared to the CFG 0.7 "
                     "original, not to preferred 04. This is the intended "
                     "comparison and that unrelated metric is not reported as "
                     "reproduction error."),
            "new_sha256": sha(new_raw),
            "byte_identical": sha(new_raw) == PRESERVED_04_RAW_SHA,
            "sample_rate_preserved": sra, "sample_rate_new": srb,
            "frames_preserved": int(len(a)), "frames_new": int(len(b)),
            "same_shape_and_rate": bool(same_shape),
        }
        if same_shape:
            difference = a - b
            comparison.update(
                max_absolute_difference=float(np.max(np.abs(difference))),
                rms_difference=float(np.sqrt(np.mean(difference ** 2))),
                rms_difference_dbfs=float(20 * np.log10(max(float(np.sqrt(np.mean(
                    difference ** 2))), 1e-30))),
                correlation=float(np.corrcoef(a, b)[0, 1]),
                preserved_peak=float(np.max(np.abs(a))),
                new_peak=float(np.max(np.abs(b))),
                preserved_rms_dbfs=float(20 * np.log10(float(np.sqrt(np.mean(a ** 2))))),
                new_rms_dbfs=float(20 * np.log10(float(np.sqrt(np.mean(b ** 2))))),
                samples_at_or_beyond_full_scale=int(np.sum(np.abs(b) >= 1.0)))

        # -- listening copy, using the preserved scalar rather than a new match
        listening = OUT / "04-reproduced-reduced-core-guidance.wav"
        scaled = b * 10 ** (PRESERVED_04_GAIN_DB / 20)
        sf.write(str(listening), scaled, SR, subtype="PCM_16")
        listening_record = {
            "path": str(listening), "sha256": sha(listening),
            "gain_db": PRESERVED_04_GAIN_DB,
            "gain_source": ("the preserved 04 scalar, reused rather than "
                            "recomputed by a fresh loudness match"),
            "subtype": "PCM_16",
            "processing": "one scalar gain only; no filtering, EQ or new target",
            "matches_preserved_listening_bytes":
                sha(listening) == PRESERVED_04_LISTENING_SHA,
            "preserved_listening_sha256": PRESERVED_04_LISTENING_SHA,
            "note": ("The historical listening RMS was -17.96973386044995 dBFS, "
                     "not exactly -18."),
        }

        for path, expected in EXPECTED.items():
            assert sha(path) == expected, f"{path} changed during the run"
        assert sha(PRESERVED_04_RAW) == PRESERVED_04_RAW_SHA
        assert sha(PRESERVED_04_LISTENING) == PRESERVED_04_LISTENING_SHA

        update(status="completed", exit_code=0,
               elapsed_seconds=time.perf_counter() - started,
               finish_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               raw_output=str(new_raw), comparison_to_preferred_04=comparison,
               listening=listening_record,
               capture_shapes={k: list(v.shape) for k, v in capture.items()},
               mel_arguments=loaded["mel_args"],
               originals_unchanged=True)
    except BaseException as exc:                    # noqa: BLE001 - recorded
        update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               exit_code=1, error=repr(exc),
               elapsed_seconds=time.perf_counter() - started)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
