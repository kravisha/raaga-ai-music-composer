"""Guide → target-conditioned audio, through the installed Seed-VC path.

Contract `46e5ca2` §2. The installed runtime, its models and its config are read
where they lie and never written to. Every mutable path — caches, working
directory, composer home, outputs — is inside the render's own directory, and
bytecode writing is off before the runtime is imported, so importing someone
else's source leaves nothing in their tree.

**Inference only.** No weights are updated, nothing is admitted to any corpus,
and the result is synthetic reference-conditioned audio.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

TRIAL = Path(r"C:\Users\Krish\ClaudeStuff\Raaga-Personal-Voice-Trial-20260908")
CHECKPOINT = (TRIAL / "models" / "Plachta--Seed-VC" /
              "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
CONFIG = TRIAL / "inputs" / "local-config.yml"

#: The settings the existing runs used, so this is comparable to them.
SETTINGS = dict(diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=.7,
                f0_condition=True, auto_f0_adjust=False, semi_tone_shift=0,
                fp16=False)

#: Seeds actually used by the two existing baselines, so a comparison can say
#: which of them it holds constant and which it does not.
BASELINE_SEEDS = {"personal_voice_trial": 20260908, "spb_conditioned": 20260909}


class ConversionError(Exception):
    """A refusal. Never a silent correction."""


class SeedVC:
    """One conversion per call, offline, on the CPU."""

    def __init__(self, *, seed: int, compared_baseline: str | None = None):
        self.seed = int(seed)
        self.compared_baseline = compared_baseline

    def __call__(self, guide: Path, reference, directory: Path, record) -> Path:
        for required in (CHECKPOINT, CONFIG, Path(reference.path)):
            if not Path(required).is_file():
                raise ConversionError(f"missing installed input: {required}")

        work = Path(directory) / "conversion-work"
        out = Path(directory) / "conversion-output"
        work.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)

        sys.dont_write_bytecode = True
        os.chdir(work)
        os.environ.update(
            CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_CACHE=str(work / "hf"), HF_HOME=str(work / "hf"),
            TORCH_HOME=str(work / "torch"), NUMBA_CACHE_DIR=str(work / "numba"),
            MPLCONFIGDIR=str(work / "matplotlib"),
            RAAGA_COMPOSER_HOME=str(work / "isolated-unused-raaga-home"))
        sys.path.insert(0, str(TRIAL / "runtime-deps"))
        sys.path.insert(0, str(TRIAL / "runtime-source"))

        import numpy as np
        import soundfile as sf
        import torch

        torch.set_num_threads(6)
        torch.set_num_interop_threads(2)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        import inference
        inference.device = torch.device("cpu")

        def local_model(repo_id, model_filename="pytorch_model.bin",
                        config_filename=None):
            folder = TRIAL / "models" / repo_id.replace("/", "--")
            model = folder / model_filename
            if not model.is_file():
                raise ConversionError(f"missing model {model}")
            if config_filename is None:
                return str(model)
            configuration = folder / config_filename
            if not configuration.is_file():
                raise ConversionError(f"missing config {configuration}")
            return str(model), str(configuration)

        inference.load_custom_model_from_hf = local_model
        written: list = []

        def save_audio(path, tensor, sample_rate, **kwargs):
            path = Path(path).resolve()
            if not path.is_relative_to(out.resolve()):
                raise ConversionError(f"{path} is outside {out}")
            if path.exists():
                raise ConversionError(f"{path} already exists")
            values = tensor.detach().cpu().numpy().T
            if not np.isfinite(values).all():
                raise ConversionError("non-finite converted audio")
            sf.write(str(path), values, sample_rate, subtype="PCM_24")
            written.append(path)

        inference.torchaudio.save = save_audio

        record.settings(backend="seed-vc", device="cpu", precision="float32",
                        seed=self.seed, training=False,
                        network="offline; pinned local models only", **SETTINGS)
        record.input("checkpoint", path=CHECKPOINT)
        record.input("conversion_config", path=CONFIG)
        if self.compared_baseline:
            theirs = BASELINE_SEEDS.get(self.compared_baseline)
            if theirs is not None and theirs != self.seed:
                record.not_constant("seed", self.seed, theirs,
                                    against=self.compared_baseline)

        inference.main(SimpleNamespace(
            source=str(guide), target=str(reference.path), output=str(out),
            checkpoint=str(CHECKPOINT), config=str(CONFIG), **SETTINGS))

        if len(written) != 1:
            raise ConversionError(f"expected one output, got {len(written)}")
        return written[0]
