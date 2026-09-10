# SPB direct TTS: independent architectural assessment and setup notes

Author: Claude Code, `claude/spb-direct-tts-20260910`, worktree
`raaga-wt-claude-spb-direct-tts`, at shared spec commit
`2443c85702998d31e40057270ccad77bd7e5485f`. This is my own assessment, not a
restatement of Arya/Codex's or conversational Claude's conclusions. Where I
agree with the option Krish already selected, I say so with my own
reasoning and my own list of what remains unproven; where I have residual
doubt, I say that too.

## The two options as I understand them

- **Option A — generic MMS + adapted voice conversion.** Keep
  `facebook/mms-tts-tam` unmodified, keep synthesizing generic Tamil speech,
  and invest further in the voice-conversion stage (better/more SPB
  reference audio, a different VC model, VC hyperparameter or checkpoint
  tuning) to push the converted output closer to SPB.
- **Option B — direct fine-tuned MMS.** Fine-tune a separate copy of the
  MMS Tamil acoustic model itself on reviewed SPB speech, so the generator
  natively produces SPB timbre and articulation, removing the VC stage
  (or reducing it to a much smaller residual role).

Krish selected B. Arya/Codex independently agrees with B. My own reasoning
below also favors B as the right next experiment, but for a narrower and
more conditional reason than "it's obviously better," and with specific,
currently unresolved feasibility risk that a synthetic-fixture pilot has
not yet tested.

## Why I agree with B, on the merits

The concrete evidence Krish reported is that the current two-stage
pipeline (unmodified MMS text-to-speech, then Seed-VC conversion toward
SPB) does add audible SPB character but leaves an unwanted accent. That
observation is doing real work here, but only for a specific reason: voice
conversion, as a class of technique, is built to transform speaker
identity/timbre on top of an already-fixed linguistic and prosodic
rendering; it is not designed to change the phonetic and prosodic choices
the source TTS already committed to when it synthesized the utterance. An
"accent" complaint is at least partly a complaint about *how* sounds are
articulated and timed, not only about *whose* spectral envelope is on top
of them. If that diagnosis is right, no amount of VC iteration on the
current MMS output has a principled reason to remove it, because VC never
touches that layer. That is a mechanistic argument for why B addresses the
actual complaint more directly than continuing to iterate on A, and it is
different from — and I think stronger than — simply noting that the VC
output "still doesn't sound like SPB enough." I want to be explicit that
this is my own inference about how VC and TTS interact, not a proven fact
about this specific Seed-VC/MMS combination; it is a reasonable prior, not
a measurement.

I want to be equally explicit about what this does **not** establish, per
the spec's own framing: it does not prove voice conversion has an absolute
ceiling. A stronger VC model, more/better SPB reference audio, or VC
applied at an earlier acoustic stage could plausibly close more of the gap
than Seed-VC did here. Replacing the TTS base is justified by "this is a
promising, currently-untested route with a principled mechanism and an
available matched checkpoint," not by "the alternative has been ruled
out." Both statements are compatible with proceeding with B first, which
is what I'd recommend independent of anyone else's agreement.

## What is unproven and should gate further investment

1. **No VITS-training memory/throughput measurement exists on this
   hardware.** The only local benchmark on record is a Seed-VC one-shot
   2-second synthetic conversion step: 2.609 s, 4.90 GiB peak allocation.
   That is an inference-style forward pass through a different model
   family; VITS training runs a generator *and* a discriminator, an
   adversarial loss, a mel-spectrogram loss, a KL term, and (in the
   upstream implementation) a monotonic alignment search, all with
   backward passes and optimizer state. None of that is captured by the
   Seed-VC number, and I would not treat 4.90 GiB as informative about
   whether a VITS training step fits in the roughly 5 GB free on this
   GTX 1060 6GB (Ryzen 7 3700X, ~32 GB system RAM). This has to be measured
   directly, at batch size 1, before any real-data pilot is worth planning
   in detail.
2. **The local MMS checkpoint is inference-only.** num_speakers=1,
   speaker_embedding_size=0, 16 kHz, and 36,288,432 saved parameters, with
   no discriminator tensors. That checkpoint cannot be a training start
   point by itself; it has no adversarial head. The plan already correctly
   points at `ylacombe/mms-tts-tam-train` as the training-ready checkpoint
   with a discriminator, but nobody has yet diffed that checkpoint's
   generator tensor keys/shapes/values against the local inference-only
   generator. The spec requires that comparison before calling the two the
   same initialization, and it is a real check, not a formality — Tamil
   MMS checkpoints under different names are not guaranteed to be
   bit-identical generators.
3. **Two dependencies are simply absent from the current inference
   environment** (`torch 2.7.1+cu126`, `transformers 4.57.6`,
   `accelerate 1.12.0`): `datasets` and `Cython`. Cython specifically
   matters because `finetune-hf-vits`'s monotonic alignment search is a
   compiled C extension; whether it builds cleanly against this Windows
   toolchain and this CUDA/torch combination is an open environment risk
   independent of the modeling question. I would treat "the MAS extension
   builds and runs a synthetic forward/backward pass" as a distinct,
   earlier gate from "training fits in 6 GB VRAM."
4. **Version drift risk between the pinned upstream code and the current
   inference stack.** `transformers 4.57.6` is recent; `finetune-hf-vits`
   was written against whatever `transformers` VITS API existed when it was
   published, and HF's VITS modeling code has changed shape over time
   (config fields, discriminator wiring). I would not assume the upstream
   training script runs unmodified against `transformers 4.57.6` without
   trying it; if it needs an older pinned `transformers`, that must be a
   dependency of the *new, separate* training environment only — the
   working inference environment must not be touched, per the spec.
5. **No approved training corpus exists yet.** Four SBS interview parts
   total 54m28.872s gross, plus a separate ~42m27.641s Sakshi-session
   source, but every clip drawn from them so far (including the two I
   produced in this session and the two Codex produced from part 3) is an
   unreviewed draft: `accepted_for_training=false`, review fields pending.
   The single 12.5-second diagnostic clip that begins exactly "அத" is
   already exposed and correctly treated by the spec as a development
   anchor, not a fresh held-out test — my planner enforces that structurally
   (see below). Until Krish reviews text/speaker/quality on a real batch
   and an independent, session-disjoint clip is similarly reviewed for
   eval, there is no dataset to fine-tune on regardless of which
   architecture is chosen, and after excluding interviewer turns, overlap,
   and music, the clean-single-speaker duration actually available could
   be meaningfully less than the gross figures above.

None of this is an argument against B. It is an argument for treating the
contract's own sequencing — synthetic-fixture forward/backward first, tiny
bounded real-data pilot second, everything gated on measured evidence — as
load-bearing rather than procedural. If the synthetic pilot OOMs on this
GPU or the MAS extension will not build here, that is a legitimate result
that should redirect effort (smaller batch/duration, gradient checkpointing
if the implementation supports it, or reconsidering scope) rather than a
reason to declare the experiment failed after one attempt.

## Setup recommendations (for whoever runs the pilot)

- Isolated environment: a new venv separate from the working inference
  environment, pinned to whatever `finetune-hf-vits` commit is used,
  recording the exact `torch`/`transformers`/`accelerate` versions that
  environment resolves to (may differ from `2.7.1+cu126` /
  `4.57.6` / `1.12.0` — record any divergence and any patch needed to
  make the pinned commit run, per the spec's compatibility-patch
  requirement). Add `datasets` and `Cython` there only.
- Build and smoke-test the monotonic alignment extension in isolation
  before touching any model code.
- Diff `ylacombe/mms-tts-tam-train`'s generator tensor keys/shapes/values
  against the local inference-only MMS Tamil checkpoint; record the result
  before calling them the same initialization.
- Override every unsafe upstream default explicitly: disable
  `push_to_hub`, do not use `overwrite_output_dir` against an existing
  directory, and use genuinely separate train/eval manifests (never the
  upstream pattern of one split serving both).
- First pilot step: synthetic tensors only, shapes matching the target
  16 kHz / mel-spectrogram configuration MMS expects (note: the review
  clips extracted so far, including mine, are 48 kHz FLOAT WAV exact
  crops of the untouched source — any resampling to 16 kHz happens only
  in a training data-loading step, never by re-processing the immutable
  review clips). Run one forward+backward pass at batch size 1, record
  `torch.cuda.max_memory_allocated()`, and stop. Do not proceed to real
  data until that number is known and is comfortably under the ~5 GB
  free budget with margin for the data loader and any held-out eval pass.
- Start FP32 as the spec requires; treat a measured FP32 OOM at batch
  size 1 as the evidence needed to justify trying FP16/mixed precision,
  and record that justification rather than switching precision by
  default.
- Keep the training venv and the working inference venv fully separate;
  no shared upgrades either direction.

## What I did not do

I did not install anything, load any model, acquire any checkpoint, or run
any training or inference in this environment. Corpus acquisition,
environment build-out, and the actual pilot remain Codex's and Krish's to
execute per the ownership split in `CLAIMS.md`; this document is my
independent read of the tradeoffs and risks, for the record.
