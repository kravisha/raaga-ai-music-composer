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
observation is doing real work here, but only for a specific reason:
**Correction to an earlier version of this section** — I originally wrote
that voice conversion "never touches" the phonetic/prosodic layer, as a
claim about VC as a general class of technique. That is too broad and I
was wrong to state it that way. Seed-VC's own V2 release
(`hubert-bsqvae-small`, invoked via `--convert-style` in
`inference_v2.py`) is explicitly documented and named "Voice & Accent
Conversion," using an ASTRAL-Quantization content encoder specifically
built to suppress source-speaker traits — i.e. a VC design that does
target accent, which is a prosodic/articulatory property, not just
timbre. So the general claim "VC as a class cannot address accent" is
false on the evidence of that source (`Raaga-Personal-Voice-Trial-20260908/source/inference_v2.py`,
`README.md`).

What I can still say with more confidence is narrower: the specific
Seed-VC configuration actually in use in this project's pipeline is a
V1/SVC-style, F0-preserving chain, not V2's accent-targeting mode. An
F0-preserving VC stage is built to carry the source utterance's pitch
contour and timing through largely unchanged while replacing spectral/
timbre content; by construction that leaves less room for it to correct
articulatory choices (segment durations, consonant realization, stress
pattern) baked in by the upstream MMS synthesis than a mode explicitly
designed to convert style/accent. If Krish's "unwanted accent" complaint
is at least partly about *how* sounds are articulated and timed rather
than only *whose* spectral envelope sits on top of them, that is a
plausible reason our specific F0-preserving chain would leave it largely
intact — but it is a claim about the chain we're actually running, not
about voice conversion in general, and switching to a V2/accent-targeting
Seed-VC mode is a genuinely different, untested option that this
correction does not rule out. I want to be explicit that this narrower
claim is still my own inference about how our specific F0-preserving
chain behaves, not a proven fact measured on this SPB/MMS combination; it
is a reasonable prior, not a measurement.

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

1. **A synthetic single-step VITS training measurement now exists, but it
   is narrower than full feasibility.** Codex's `runtime-probe-01/result.json`
   (not something I ran; I'm reporting it, not repeating it) records one
   complete VITS generator+discriminator forward/backward/optimizer step
   on a 1-second synthetic tone, FP32, batch size 1, on this GTX 1060 6GB:
   2.587 s wall time, 1,899,811,328 bytes (≈1.77 GiB) peak GPU allocated,
   744/111 Adam optimizer-state entries for generator/discriminator,
   finite non-NaN losses (generator 66.72, discriminator 7.74), and the
   checkpoint file left unchanged on disk (no learned checkpoint was
   saved — this was a diagnostic step, not training). That is a real,
   materially stronger result than the earlier Seed-VC-derived 4.90 GiB
   placeholder I flagged before: it is an actual VITS train step, and its
   peak allocation leaves comfortable headroom under the ~5 GB budget at
   this batch size. It still does **not** establish long-clip feasibility,
   multi-step/multi-epoch memory growth, a real data loader, or SPB audio
   at all — the probe's own `limits` field says so explicitly (one 1-second
   synthetic signal, not speech training or quality evidence). I'd treat
   this as "the minimal case does not immediately OOM," not "training on
   real SPB clips is proven to fit."
2. **The generator-tensor diff I called for now exists and resolves the
   specific risk I flagged.** `assets/initialization-comparison.json`
   shows all 762 generator tensors in the downloaded training checkpoint
   (`mms-tts-tam-train-481128b97cd2`) match the local inference-only
   `facebook/mms-tts-tam` generator's tensor keys/shapes/values exactly
   (`generator_byte_identical: true`, zero differences), with 74
   additional tensors layered on top, all under the `discriminator` group
   and nothing else unexplained. The recorded config differences
   (`discriminator_periods`, `segment_size`, `VitsModelForPreTraining` vs
   `VitsModel`, a newer `transformers_version` string at export time) are
   exactly what a training-ready wrapper around the same generator should
   add, not evidence of a different generator. This closes the "Tamil MMS
   checkpoints under different names are not guaranteed to be
   bit-identical" risk for this specific pair — I did not run this
   comparison myself, I'm reading Codex's recorded artifact.
3. **The Cython/MAS build gate has been exercised, with a disclosed
   substitution rather than a Cython success.** `finetune-hf-vits`'s
   monotonic alignment search is normally a compiled Cython/C extension;
   per `runtime-probe-01/result.json`, the actual MSVC build was attempted
   and failed (no MSVC toolchain present), and an explicit Numba-based
   adapter was substituted and verified to match the upstream Python
   alignment recurrence exactly on 18 fixture cases. That is a legitimate,
   recorded resolution of the environment risk I called out, not a report
   that the original Cython path works — the spec's compatibility-patch
   allowance covers a disclosed, verified-equivalent substitute, and this
   is disclosed here and in the artifact itself. It does not establish
   alignment behavior on alignment shapes outside those 18 fixtures, and
   it must stay confined to the new, separate training environment, not
   the working inference environment.
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
   be meaningfully less than the gross figures above. As of this update,
   Krish is actively dictating corrections for one clip
   (`cc-sbs2-060000`) through the coordinator; its own record
   (`data/review-batch-01/human-corrections-in-progress.json`) still
   reads `status: in_progress_not_approved` and
   `training_admitted: false`, and a prior bounded ASR retry on this same
   clip was explicitly rejected as making things worse. I have not
   touched that file, the manifest, or any clip text — this remains
   Krish's and Codex's review to finish, not mine.

None of this is an argument against B. It is an argument for treating the
contract's own sequencing — synthetic-fixture forward/backward first, tiny
bounded real-data pilot second, everything gated on measured evidence — as
load-bearing rather than procedural. If the synthetic pilot OOMs on this
GPU or the MAS extension will not build here, that is a legitimate result
that should redirect effort (smaller batch/duration, gradient checkpointing
if the implementation supports it, or reconsidering scope) rather than a
reason to declare the experiment failed after one attempt.

## Setup recommendations (for whoever runs the pilot)

Three of the items below are now done, per Codex's recorded artifacts
(`runtime-probe-01/result.json`, `assets/initialization-comparison.json`)
discussed above; I list them here for completeness and to record that I
am not re-running them, not as open work:

- ~~Build and smoke-test the monotonic alignment extension in isolation
  before touching any model code.~~ Done, via a disclosed Numba
  substitute after the native MSVC build failed (item 3 above).
- ~~Diff `ylacombe/mms-tts-tam-train`'s generator tensor keys/shapes/values
  against the local inference-only MMS Tamil checkpoint; record the result
  before calling them the same initialization.~~ Done: byte-identical
  generator, 74 added discriminator tensors (item 2 above).
- ~~First pilot step: synthetic tensors only... run one forward+backward
  pass at batch size 1, record peak memory, and stop.~~ Done: 2.587 s,
  ≈1.77 GiB peak allocated, FP32, batch 1, 1-second synthetic tone (item
  1 above). Still open: the same measurement on a real, multi-second
  Tamil clip and across more than one step.

Still open:

- Isolated environment: a new venv separate from the working inference
  environment, pinned to whatever `finetune-hf-vits` commit is used,
  recording the exact `torch`/`transformers`/`accelerate` versions that
  environment resolves to (may differ from `2.7.1+cu126` /
  `4.57.6` / `1.12.0` — record any divergence and any patch needed to
  make the pinned commit run, per the spec's compatibility-patch
  requirement). Add `datasets` and `Cython` there only.
- Override every unsafe upstream default explicitly: disable
  `push_to_hub`, do not use `overwrite_output_dir` against an existing
  directory, and use genuinely separate train/eval manifests (never the
  upstream pattern of one split serving both).
- Next pilot step, building on the completed 1-second synthetic
  measurement above: shapes matching the target 16 kHz / mel-spectrogram
  configuration MMS expects on a real, multi-second duration and across
  more than one step (note: the review clips extracted so far, including
  mine, are 48 kHz FLOAT WAV exact crops of the untouched source — any
  resampling to 16 kHz happens only in a training data-loading step,
  never by re-processing the immutable review clips). Do not proceed to
  real SPB data until that number is known and is comfortably under the
  ~5 GB free budget with margin for the data loader and any held-out
  eval pass.
- Start FP32 as the spec requires; treat a measured FP32 OOM at batch
  size 1 as the evidence needed to justify trying FP16/mixed precision,
  and record that justification rather than switching precision by
  default.
- Keep the training venv and the working inference venv fully separate;
  no shared upgrades either direction.

## What I did not do

I did not install anything, load any model, acquire any checkpoint, or run
any training or inference in this environment. I did not re-run or repeat
Codex's synthetic training-step probe or checkpoint diff; the measured
facts I cite above are read from `runtime-probe-01/result.json` and
`assets/initialization-comparison.json`, not reproduced by me. Corpus
acquisition, environment build-out, and the actual pilot remain Codex's
and Krish's to execute per the ownership split in `CLAIMS.md`; this
document is my independent read of the tradeoffs and risks, for the
record.

## Update (this turn): manifest/planner hardening

Following an independent external review of `direct_tts/manifest.py` and
`direct_tts/planner.py` (`claude-validator-independent-findings.json`,
`Raaga-SPB-Architecture-Review-20260910`), I fixed five concrete gaps and
added regression tests for each, in `direct_tts/` and
`tests/test_direct_tts_manifest.py` only:

1. **Content-identity overlap bypass.** Cross-entry overlap/duplicate
   detection keyed on `source_path` (a string label), so two entries with
   the same audio bytes under different paths and different
   `source_group` labels could both be admitted as if independent.
   `find_cross_entry_issues` now keys interval-overlap detection on
   `source_sha256` (content identity), and `validate_entry` now requires
   a nonempty `source_group` on every entry — a missing group is never
   treated as evidence of independence.
2. **Path traversal via `id`.** An entry with `id="../../escape"` produced
   an export path that escaped the requested root
   (`dataset/train/../../escape.wav`). `validate_entry` now restricts
   `id` to a safe filename-stem pattern (no path separators, no leading
   dot), and `plan_export` independently verifies every computed export
   path resolves inside the requested root before returning it — defense
   in depth, not reliance on id validation alone. `plan_export` also now
   refuses to plan into an export root that already exists on disk
   (checked via the same injectable `exists_fn`), unless the caller
   explicitly passes `dry_run_view=True` for a preview-only use.
3. **`end_seconds=inf` and boolean start/end passed validation.** `bool`
   is a subclass of `int` in Python, so `True`/`False` satisfied the old
   `isinstance(..., (int, float))` check, and `inf` satisfied `end > start`.
   `validate_entry` now requires finite, non-boolean numeric boundaries.
4. **A metadata-only plan reported `ready=True` with no filesystem check.**
   The plan only ever reasoned about manifest dict fields; it never
   confirmed the source/clip files exist or that their bytes still match
   the recorded hash. The field is renamed `metadata_ready` to say
   honestly what it checks (I found no other consumer of the old
   `ready` name to preserve compatibility for). A new module,
   `direct_tts/verifier.py`, adds the missing filesystem-backed step:
   `verify_plan()` re-confirms root-bound export paths, source/clip
   existence, and source/clip SHA-256 match, via an injectable digest
   function so tests use plain temp files, never real audio.
   `execution_ready` in its result is the only signal meaning "safe to
   actually export." I also found that the emitted `TrainEntryPlan`/
   `EvalEntryPlan` dropped `source_sha256` and the reviewed clip's
   path/hash entirely; both are now retained on every emitted plan entry.
5. **No supported path for an explicitly bounded within-session eval
   pilot.** The planner only supported session-disjoint eval, with no way
   to explicitly and truthfully run a bounded engineering pilot with
   within-session eval. `plan_export` now takes `eval_policy`, defaulting
   to `"session_disjoint"` (unchanged behavior), with an explicit
   `"within_session_blocked"` option for that narrower case. Regardless of
   policy, each eval entry's `eval_kind` always reflects its actual
   measured session relationship — the policy only decides whether a
   within-session entry can satisfy readiness, never whether it gets
   relabelled as session-disjoint. Nonempty-split and source-overlap
   rejection are unaffected; `review_only` entries are still excluded, not
   rejected, under both policies.

I revised the existing test suite in the same file: `plan.ready` became
`plan.metadata_ready` throughout (an intentional rename per point 4, not
a silent behavior change), and `_train_eval_fixture()`'s eval entry was
given its own distinct `source_sha256` — the previous fixture
accidentally gave train and eval entries the same default source hash for
two nominally different sessions, which the old path-keyed conflict logic
missed but the new hash-keyed logic would (correctly) now reject; that
was a latent gap in the fixture, not a spec change. All 33 pre-existing
tests still pass under their original intent (with that one fixture
correction and the `ready`→`metadata_ready` rename); I added 22 new tests
(55 total, all passing) covering the five findings above.

I also corrected an over-broad claim in the "Why I agree with B" section
above: I had written that voice conversion as a class never touches
phonetic/prosodic choices. Seed-VC's own V2 release is explicitly named
"Voice & Accent Conversion" and built to suppress source-speaker traits,
which falsifies that general claim. I narrowed it to our actual
F0-preserving V1/SVC chain, where the reasoning still plausibly applies;
see that section for the corrected argument. This does not change my
agreement with option B.
