# Direct SPB Tamil TTS adaptation — decision and execution contract

Revision 1, September 10, 2026. Krish selected option B: adapt text-to-speech itself to SPB, initially by fine-tuning a separate copy of the existing MMS Tamil model. This is a scoped addition to `SPB_VOICE_TRAINING_WORKFLOW.md`; the broader goal remains natural singing of new Tamil lyrics. Speech success alone does not establish singing or score control.

## Decision and evidence

The approved plain `facebook/mms-tts-tam` output preserved the words well enough for Krish to proceed. The subsequent established Seed-VC conversion added some SPB character, while Krish heard an unwanted accent remain. That observation motivates direct TTS adaptation; it does not prove an absolute ceiling in downstream conversion. Fine-tuning the current TTS is a route to direct SPB TTS, not a competing third option. Replacement of the TTS base needs concrete comparative evidence.

Arya/Codex concurs with this direction. Krish relayed conversational Claude's recommendation for direct adaptation in a separate copy; this is user-relayed advice, not direct tool contact. Claude Code is a separate coding collaborator. Krish explicitly authorized starting without waiting for its concurrence.

## Ownership and common revision

Canonical specification namespace is this repository's `docs/spec`, indexed by `CROSSREF.md`. Both developers start from this identical committed revision in separate branches and worktrees. The existing main and voice-prototype checkout remain untouched; no merge or push is implied. Codex owns the corpus manifest, asset acquisition, isolated environment, execution and independent review. Claude Code owns only the newly assigned `direct_tts/`, focused tests and setup report in its separate worktree. No duplicate model or GPU jobs. Record the shared spec commit in both run records and the handoff.

All work is headless. No computer, mouse, native-app or browser input. No paid cloud/GPU purchase. All original recordings, model weights, approved text and earlier outputs are immutable.

## Data contract

1. Source records contain original path/hash, provenance, session group and exact clip bounds. No transcript or speaker identity is inferred solely from a filename or ASR output.
2. Start with clean SPB speech. Exclude interviewer, overlapping speakers, music, clipped phonemes and heavily processed/separated material from the initial training batch. Keep singing recordings in a separately labeled inventory for later singing work; do not pool speech, Krish's voice, interviewer speech or song accompaniment under one target identity.
3. Every training clip requires reviewed text, reviewed target-speaker/quality status and recorded use scope. Krish has authorized the existing recordings for this local experiment; retain provenance and any recorded external use constraints without presenting them as a new request for routine permission. No public distribution is included.
4. Machine transcripts are drafts. The approved diagnostic starts exactly **அத**, not அதை or அது. Preserve colloquialisms and contrastive Tamil sounds. Reject empty, disputed or unreviewed labels before training.
5. Split before training. Keep all four SBS interview parts in one session group. Use a different verified session for independent-source evaluation when available; a blocked within-session split is labeled accordingly. No overlapping/adjacent duplicate source windows across splits. Keep the already exposed 12.5-second diagnostic as an anchor/development example, never a fresh blind test.
6. Draft review clips live outside training/evaluation input directories. Only an explicit admitted manifest can populate a training dataset. Store sentence text, uncertainty, reviewer and source positions together.

## Training and environment

Use an isolated environment and pinned upstream training source, a training-ready Tamil MMS checkpoint with discriminator, and the unchanged existing generator as the comparison. Compare generator tensor keys/shapes/values before calling a training checkpoint the same initialization. Do not infer training readiness from an inference-only load. Pin dependencies and record any compatibility patches; do not modify the working inference environment.

A pilot requires tested manifest guards, matching sample rate/tokenizer, finite losses, reproducible seed, nonexisting output directory, a hard step/time bound, owned process termination and no implicit Hub upload. Disable experiment-service uploads. Separate train and evaluation inputs, including when the upstream example uses one split for both. Do not use upstream overwrite-output defaults.

First validate shapes/forward/backward and resource use with synthetic fixtures only, then a bounded pilot on reviewed real data. A one-utterance overfit smoke test, if used, is labeled engineering validation and not SPB quality or generalization. Do not silently turn all source audio into training data to meet a nominal duration.

The measured local hardware is GTX 1060 6GB, Ryzen 7 3700X, approximately 32GB RAM. No MMS training memory/throughput measurement yet exists. A prior Seed-VC synthetic two-second step consumed about 4.90GiB peak allocation and took 2.609 seconds; it is not a VITS training estimate. Use FP32 initially on this device unless measured evidence supports another precision. Stop on OOM, nonfinite loss, dataset validation failure, source/hash drift or budget expiry. Preserve failed-run evidence.

## Evaluation and stop rules

Compare unchanged MMS, adapted direct TTS and the preserved two-stage baseline on the same new Tamil sentences, with fixed text, seed policy and level-matched listening copies. Include vowel length, ழ/ள/ல distinctions, nasal and V articulation, ordinary unseen sentences and held-out original speech. Report word accuracy, naturalness, vocal character/accent and artifacts separately. ASR and speaker embeddings are diagnostics, not acceptance; unfamiliar ASR errors do not rewrite Krish's approved text. The original recording is a perceptual target, not an expected bit-identical waveform reconstruction from lossy text.

Start with a bounded pilot after data and hardware readiness; use validation loss only to flag instability, and require Krish's listening before adoption. Preserve the best baseline and stop repeating unchanged settings when results flatten. Singing requires a later, explicit lyric/score-to-performance evaluation; do not obtain apparent score compliance by destroying word intelligibility.
