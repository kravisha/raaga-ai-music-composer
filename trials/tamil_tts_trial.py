"""Stage 1 — the exact Tamil words, spoken, with the model's own durations.

`facebook/mms-tts-tam` at pinned revision `e9cf59da`, a 16 kHz **speech** model,
not a singing model. Run once with a fixed seed.

The durations are taken from the model's own duration predictor rather than
invented: `VitsModel` exposes `log_duration` inside its forward pass, and the
predicted durations are what the length regulator uses to place each input
token. That gives per-character timing **as the model actually spoke it**.

Stated plainly, because it matters downstream: those are **character** timings
from a tokenizer with `phonemize=false` and `is_uroman=false`. They are not
validated phonemes, and this file does not claim they are.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
MODEL = WORKSPACE / "models" / "facebook--mms-tts-tam"
OUT = WORKSPACE / "renders" / "tamil-tts-trial"

#: Exactly as Krish wrote it. Recorded here so any normalisation is visible.
LYRICS = "நிலவாய் நீ வந்தால், இரவும் பகல் ஆகும்"
#: The comma is punctuation, not a word. Removing it is the only change, and the
#: original stays above.
LYRICS_FOR_SYNTHESIS = LYRICS.replace(",", "")
WORDS = LYRICS_FOR_SYNTHESIS.split()

SEED = 20260909


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "tts-status.json"
    report = {
        "stage": "tts",
        "pid": os.getpid(),
        "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "facebook/mms-tts-tam",
        "revision": "e9cf59dae34f0f51e3b1842876a658e4516f9fe4",
        "model_is": "16 kHz speech synthesis, NOT a singing model",
        "license": "cc-by-nc-4.0; isolated private trial, no distribution",
        "lyrics_original": LYRICS,
        "lyrics_for_synthesis": LYRICS_FOR_SYNTHESIS,
        "normalisation": "the comma was removed; nothing else was changed",
        "words_expected": WORDS,
        "seed": SEED,
        "training": False, "network_used_for_synthesis": False,
    }

    def update(**fields):
        report.update(fields)
        status.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(json.dumps(fields, default=str, ensure_ascii=False), flush=True)

    update()
    started = time.perf_counter()
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HOME=str(OUT / "cache"), TORCH_HOME=str(OUT / "cache"),
                          PYTHONDONTWRITEBYTECODE="1")
        sys.dont_write_bytecode = True

        import numpy as np
        import soundfile as sf
        import torch
        from transformers import VitsModel, VitsTokenizer

        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        tokenizer = VitsTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        model = VitsModel.from_pretrained(str(MODEL), local_files_only=True)
        model.eval()
        update(status="model_loaded",
               is_uroman=bool(getattr(tokenizer, "is_uroman", False)),
               phonemize=bool(getattr(tokenizer, "phonemize", False)),
               sampling_rate=int(model.config.sampling_rate),
               speaking_rate=float(model.config.speaking_rate),
               noise_scale=float(model.config.noise_scale),
               noise_scale_duration=float(model.config.noise_scale_duration))

        inputs = tokenizer(LYRICS_FOR_SYNTHESIS, return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # The model's own durations. `output_attentions` is not what carries
        # them; the duration predictor's output is, so it is captured directly
        # off the module rather than reconstructed.
        captured = {}
        predictor = model.text_encoder if hasattr(model, "text_encoder") else None
        original_forward = model.duration_predictor.forward

        def capture(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            captured["log_duration"] = result.detach().cpu().numpy()
            return result

        model.duration_predictor.forward = capture
        with torch.no_grad():
            output = model(**inputs)
        model.duration_predictor.forward = original_forward

        waveform = output.waveform[0].cpu().numpy().astype("float64")
        rate = int(model.config.sampling_rate)
        raw = OUT / "01-tts-raw.wav"
        sf.write(str(raw), waveform, rate, subtype="FLOAT")

        timing = None
        if "log_duration" in captured:
            log_duration = captured["log_duration"]
            # The model scales predicted durations by 1 / speaking_rate and
            # rounds up, exactly as its length regulator does.
            length_scale = 1.0 / float(model.config.speaking_rate)
            frames = np.ceil(np.exp(log_duration) * length_scale)
            frames = np.maximum(frames, 0).reshape(-1)
            # Each frame is one hop of the decoder's upsampling stack.
            hop = int(np.prod(model.config.upsample_rates)) if hasattr(
                model.config, "upsample_rates") else 256
            seconds = frames * hop / rate
            edges = np.concatenate([[0.0], np.cumsum(seconds)])
            timing = [{"index": i, "token": tokens[i] if i < len(tokens) else "?",
                       "start_s": round(float(edges[i]), 6),
                       "end_s": round(float(edges[i + 1]), 6),
                       "seconds": round(float(seconds[i]), 6)}
                      for i in range(len(seconds))]
            (OUT / "02-token-timing.json").write_text(
                json.dumps({
                    "source": ("VitsModel duration_predictor output, scaled by "
                               "1/speaking_rate and ceil-rounded as the model's "
                               "own length regulator does"),
                    "caveat": ("These are CHARACTER timings. The tokenizer has "
                               "phonemize=false and is_uroman=false, so they are "
                               "not validated phonemes and are not claimed to be."),
                    "hop_samples": hop, "sample_rate": rate,
                    "total_seconds": round(float(edges[-1]), 6),
                    "tokens": timing}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")

        update(status="completed", exit_code=0,
               elapsed_seconds=time.perf_counter() - started,
               finish_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               raw_tts={"path": str(raw), "sha256": sha(raw),
                        "sample_rate": rate, "frames": int(len(waveform)),
                        "seconds": round(len(waveform) / rate, 6),
                        "peak": float(np.max(np.abs(waveform))),
                        "rms_dbfs": float(20 * np.log10(float(np.sqrt(np.mean(
                            waveform ** 2)))))},
               tokens=len(tokens),
               token_timing_captured=timing is not None,
               token_timing_total_seconds=(round(float(np.sum(
                   [t["seconds"] for t in timing])), 6) if timing else None))
    except BaseException as exc:                      # noqa: BLE001
        update(status="failed", exit_code=1, error=repr(exc),
               elapsed_seconds=time.perf_counter() - started)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
