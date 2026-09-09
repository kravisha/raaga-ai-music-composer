# SPB Voice Training Workflow — Executable Specification

**Status:** draft for reviewer concurrence. Linked to `CANONICAL_SPEC_v0.3.txt`
(SHA-256 `47f6c8a26dd1cb46fd4251ece7fe988438eed11594aca01b53cc4f01204f8c60`) and
indexed in `CROSSREF.md`. This document does not change any existing product
requirement; it adds the workflow by which a target singing voice is prepared,
trained, evaluated and — only if it earns it — adopted.

**Base revision:** `c36dc2207d939d0cb7b47575298f445d8fc09bc9`, branch
`claude/spb-training-spec`, a versioned replica of the canonical spec folder.

**Roles.** Claude leads implementation, refines the approach, forms hypotheses
and proposes exact file claims per stage. Codex/Raaga Vocal Preview guides,
reviews evidence and accepts stages, and is the **single execution owner** for
jobs that touch models, GPU or audio — one owner at a time, coordinated with the
audio task. Krish supplies every perceptual verdict and every source-use
decision.

---

## 1. Objective, and what would count as achieving it

**End to end: new lyrics become naturally sung audio, ultimately in SPB's
voice.** Krish writes or Raaga generates words that have never been sung; the
system produces a performance of them — correct Tamil pronunciation, the
intended pitch and timing, and expression — with SPB's identity, integrated into
a real song.

Real-singing training is expected to improve pronunciation, pitch, timing and
expression. That is the point of the training work, not merely timbre.

**Two things do not constitute fulfilment**, and the workflow must not be able to
claim success while either is true:

1. **Conversion-only fine-tuning.** Improving how well a guide is repainted does
   not produce a sung performance of new words.
2. **Requiring Krish to record every new phrase.** A system that needs a human
   guide take per phrase has not turned lyrics into singing; it has re-voiced a
   human performance.

Success is established by **Krish's judgement on unseen lyrics and on real
integrated song audio** — not by test counts, validation loss, ASR agreement or
pitch statistics. Those are diagnostics that can fail a candidate; none of them
can pass one.

### The capability gap, stated plainly

The installed path is **voice conversion**. It requires a guide performance for
every phrase and repaints its timbre. It has no component that turns lyrics and
notes into a performance, and **no amount of fine-tuning it will produce one** —
more training improves the repainting, never the ability to sing new words
unaided. Closing that gap is stage **S6**, and it is a genuine addition to the
system rather than a tuning exercise.

Until S6 delivers, every conversion depends on a recorded guide. **That
dependence is interim and must be labelled as such wherever a result is
reported** — an S1–S5 result is an identity and quality result, not evidence
that new lyrics can be sung.

### Non-goals, and what is deliberately left open

**Fixed:**

- **No personal-voice substitution.** Krish's recordings are guide/source
  performance and control material. They are never SPB target-speaker training
  data, and no stage may reclassify them as such.

**Open, and explicitly not restricted.** New models, new training jobs and
different methods are authorized where they are justified toward new lyrics
becoming natural SPB singing. Reuse of the installed Seed-VC runner, the accepted
preflight and the existing conversion path is a **preference**, because those are
built and reviewed — not a rule. If a different architecture serves §1 better,
propose it with the evidence and change course.

Two things still need their own basis regardless of this openness: **source
eligibility** (nothing admits data), and **paid spend** (nothing authorizes
money). Vocoder training likewise remains a separate objective with its own
scope, since the current runner updates the CFM and length regulator only.

---

## 2. Vocabulary, fixed for this document

| Term | Meaning |
| --- | --- |
| **Target material** | Audio *of SPB*, used to teach the model his voice. |
| **Guide / source performance** | A recording whose phrasing, timing and pitch contour drive a conversion. Krish's takes. Never target material. |
| **Reference** | The SPB audio conditioning a single conversion at inference time. |
| **Pretrained base** | The distributed Seed-VC checkpoint before any of our updates. Not "untrained". |
| **Source family** | Every file, excerpt and duplicate originating in one recording session or performance. The four SBS interview parts are **one** family. |
| **Admitted** | A source or clip that has passed §4 review and carries a recorded decision. Nothing is admitted by being downloaded, found, or named. |

---

## 3. Stages, gates and ownership

Each stage produces reviewable artifacts and ends at a gate. A gate is opened by
the named reviewer, never by the implementer, and never by this document.

| Stage | Deliverable | Implements | Reviews | Gate to pass |
| --- | --- | --- | --- | --- |
| **S1** | Corrected runner | Claude | Codex (separate worktree) | G1 |
| **S2** | Source inventory and ingestion | Claude | Codex | G2 |
| **S3** | Clip preparation, splits, manifests | Claude | Codex | G3 |
| **S4** | Training, checkpointing, stop rules | Claude | Codex (evidence), Codex (execution) | G4 |
| **S5** | Held-out evaluation and auditions | Claude | Codex (packaging), **Krish (verdict)** | G5 |
| **S6** | Score-to-performance: lyrics + melody → sung guide | Claude | Codex, **Krish (verdict)** | G6 |
| **S7** | End-to-end integration in Raaga | Claude | Codex, **Krish (verdict)** | G7 |

Stages may be implemented in parallel where their claims do not overlap, but
gates open in order: no S4 execution before G1, G2 and G3 are open.

**Two tracks.** S1–S5 are the **identity** track: they make a converted voice
sound like SPB and sound natural. S6 is the **performance** track: it removes
the recorded-guide dependence. Neither track alone fulfils §1. They meet at S7.

### S1 — Correct the runner

Correct the defects reproduced in
`SeedVC-Pilot-Runner-Independent-Review-20260909/review_evidence/REVIEW.md`.
Reuse the accepted preflight `831e045` unchanged unless a specific new defect
warrants a separately claimed change.

Required behaviours: bind the verified runtime-deps path so a real trainer import
resolves Munch without reinstalling anything; resolve assets by exact
repository, relative filename and role rather than basename; refuse committed
versus current identity mismatches; reject a non-finite loss **before** counting
or publishing a step; fail the run when a required save fails; never publish
state that a failed step may have half-mutated; recognise
`torch.cuda.OutOfMemoryError`, which derives from `RuntimeError` and not
`MemoryError`, without allocating GPU memory in tests; and separate the
structured result channel from ordinary trainer stdout.

**Withdrawn:** saving after an out-of-memory stop merely because an earlier save
succeeded. Preserve the last committed pair unless failed-step state has been
verifiably rolled back.

**G1 evidence:** the reviewer's reproduced failure cases now fail closed, the
valid fixture control still passes, focused fixture tests only, no model, GPU,
training or data admission.

### S2 — Source inventory and ingestion

**Start with what already exists.** Krish has many recordings on local storage,
and Raaga Vocal Preview is assessing the known local inventories for useful
singing, identity, coverage, preparation needs and source-family grouping. That
assessment is the first input to this stage — searching for new material comes
after it, not before. Originals are preserved untouched.

Every raw file — existing or future video/audio — enters through one path and
lands in an immutable store with a **quarantine** default of not admitted.

**Three outcomes, kept distinct.** A source is `admitted`, `review_pending`
(plausibly useful, something specific still unresolved — name it), or `unusable`
(with the reason recorded). Collapsing the middle case into either neighbour is
how good material gets discarded and unreviewed material gets used.

**`source_record` schema** (one per raw file):

```jsonc
{
  "source_id": "stable identifier",
  "path": "...", "sha256": "...", "bytes": 0, "added_utc": "...",
  "source_family": "one session or performance; siblings share it",
  "provenance": { "origin": "...", "acquired_utc": "...", "acquired_by": "..." },
  "role": "target | guide | control",          // SPB, Krish, or diagnostic
  "content": "singing | speech | mixed | unknown",
  "identity": { "claimed": "SPB", "verified": false, "evidence": null,
                "verified_by": null },
  "isolation": { "solo": false, "accompaniment": "unreviewed",
                 "other_speakers": "unreviewed" },
  "condition": { "recording_notes": "...", "era": "...", "processing_history": [] },
  "source_use": { "status": "unresolved | approved | refused",
                  "basis": null, "evidence": null, "decided_by": null },
  "usable_intervals": [], "rejected_intervals": [{ "range": [], "reason": "" }],
  "admitted": false
}
```

Rules: raw files are never modified. `role: target` and `role: guide` are
disjoint and a record cannot change role. Admission requires `identity.verified`,
a reviewed `isolation`, `source_use.status == "approved"` with evidence, and at
least one usable interval. **Downloading, finding or naming a file admits
nothing.**

**G2 evidence:** the store round-trips; a source missing any admission
prerequisite stays quarantined; the four SBS parts resolve to one family; a
guide record cannot be promoted to target.

### S3 — Clips, splits and manifests

**`clip_record`:** `clip_id`, parent `source_id` and `source_family`, interval in
the unchanged original, `sha256`, measured duration, `content`, reviewer,
`split`.

**Split rules.** Assign by **source family**, never by clip. Three disjoint
allocations: **train**, **validation** (used repeatedly to choose checkpoints)
and a **reserved test** (touched once, after selection). The reserved test lives
outside every training manifest and directory, and its identities are verified
before use. A ~10% validation allocation is planning guidance; the number of
independent recordings and the coverage they provide matter more.

**Leakage controls, enforced not assumed.** No family on two sides. No duplicate
content across sides, by hash and by file identity — a clip copied to a new path
with a new name is the same clip. No derivative of a training clip in
validation or test. The dataset directory contains exactly the training clips
and nothing else, because the loader walks it.

**Coverage is recorded, not assumed.** Speech and singing minutes are counted
separately and never treated as interchangeable. Vowel, register and dynamic
coverage is recorded so gaps are visible before training rather than audible
after it.

**`dataset_manifest`:** the admitted clip list, per-side family sets, counted
coverage, and the hashes of every input it was derived from. Immutable once
written.

**G3 evidence:** independent reconstruction of the splits from the manifest;
attempted leakage in each form is refused; coverage counts match the clips.

### S4 — Training

Driven by the corrected runner with the accepted preflight and the owner's
ten-asset contract.

Freeze and record for every run: pretrained base checkpoint, installed code
bindings, mel and vocoder configuration, asset identities, seed, and the
`dataset_manifest` hash. A run whose bindings do not match its manifest does not
start.

**`run_manifest`** records all of the above plus the wall-clock budget and the
tested hardware envelope, stated as measured: **one full FP32 synthetic step at
2 seconds, batch 1, on the GTX 1060 — 2.609 s, 4.90 GiB peak allocated.** That is
a measured starting point. It is **not** a maximum clip length, and neither an
hour of stable training nor a successful large checkpoint save has been
demonstrated.

**Checkpointing.** Checkpoint and metadata are committed as one complete pair;
recovery ignores incomplete pairs and never deletes the last committed one.
Metadata carries optimizer, scheduler, step, epoch, RNG, data position and the
bound identities, and reports **exact** versus **approximate** restoration
honestly — field presence is not proof of exact restore, since the installed
optimizer wrappers can swallow failures.

**Stop rules, fixed before the run:** user cancellation; the wall-clock budget,
which includes checkpoint and evaluation overhead; a non-finite update; a failed
checkpoint commitment; resource failure; or a predefined lack of improvement on
validation auditions. Keep an evaluated best checkpoint and a recovery
checkpoint. **Never publish the latest merely because loss fell.**

**G4 evidence:** a short real-data pilot at the tested setting with measured
memory and time, a verified save, reload and resume, finite updates throughout,
and accurate resume classification. Only then extend within the up-to-one-hour
budget. One execution owner at a time; no paid compute assumed.

### S5 — Evaluation and auditions

**Render and preserve the pretrained-base baseline before any update.** Without
it, no later comparison can establish anything.

Fix and record: evaluation guide phrases, held-out SPB references, seeds,
conversion settings, pitch and timing targets, output format. Any deliberate
change to these creates a **new comparison**, not another checkpoint trial.

**`audition_package`:** level-matched segments in a stated order, **masked and
randomised labels** with a retained answer key, repeated pairs, and an explicit
**no-preference** option. Krish is asked separately about Tamil words,
melody and timing, SPB resemblance, and naturalness or artifacts — four
questions, because a single forced choice conflates them and one forced choice
is not a reliable improvement.

Upstream Seed-VC documents that improved speaker resemblance can trade against
word recognition. Intelligibility and resemblance are therefore judged
separately and may move in opposite directions.

**G5 evidence:** the reserved test is run only after selection; integration is
claimed only when the real Raaga path loads the chosen checkpoint and produces
audio Krish has reviewed. Technical checks are necessary and never sufficient.

### S6 — Score to performance: lyrics and melody into a sung guide

**The purpose is to stop needing a recorded guide.** Input: fitted Tamil lyrics
plus note and timing targets, both of which Raaga already produces. Output: a
sung performance of those words at those pitches, good enough to drive
conversion — or, later, good enough on its own.

**Current code gap, stated exactly.** Raaga has the *score* side: lyric
generation, lyric-to-melody fitting, section structure, note and duration
targets, and the conversion path. It has **no acoustic model that renders
lyrics and notes as sung audio**. The Seed-VC runner cannot be trained into one:
it consumes a guide's content and pitch representations and produces a mel
conditioned on a reference. There is no text or phoneme input anywhere in that
path. This is an added component, not a tuned one.

**Proposed architecture, chosen to preserve current work.** A **synthetic guide**
replaces the recorded guide, and everything downstream stays as built:

```
lyrics + notes  →  [S6 guide synthesiser]  →  sung guide
                                                  ↓
                        existing conversion (S1–S5, SPB reference)  →  SPB performance
```

Every artifact already accepted — preflight, runner, asset contract, reference
conditioning, evaluation harness — continues to apply unchanged. Only the origin
of the guide changes, from a microphone to a synthesiser. If the S6 output later
becomes good enough in SPB's voice directly, the conversion step can be
revisited; that is not assumed.

**The alternative** — training a full singing synthesiser on SPB directly, so
lyrics and notes produce SPB audio in one model — is recorded as an option and
**not selected**. It needs aligned SPB singing with lyrics and note annotations,
far beyond the currently admitted data (which is zero), and it discards the
working conversion path.

**Requirements for the guide synthesiser.**

| Requirement | Why | Status |
| --- | --- | --- |
| Tamil phoneme coverage | The words are Tamil. Most available singing synthesisers target Japanese, Mandarin or English. | **Unresolved and the main risk** |
| Note and duration input | Must sing Raaga's melody, not its own | Raaga supplies both today |
| Stable, natural phrasing | Conversion inherits prosody: a stiff guide gives a stiff result, exactly as today's PSOLA guide does | Unknown until heard |
| Deterministic and offline | Same constraints as every other stage | Required |
| No lossy post-processing | Today's ring came from exactly that | Required |

**Data requirements.** A voicebank or model that sings Tamil. Whether this is
sourced, adapted from a multilingual model, or built from recordings is an open
decision. If it is built, its training data is **guide-role** material and is
governed by §2 and §4 like any other source — and it is still not SPB target
data.

**G6 evidence:** unseen lyrics — words never sung in any training or tuning
material — rendered to a guide and then converted, in a masked level-matched
audition. Krish judges Tamil pronunciation, melody and timing, expression, and
naturalness separately. A guide that is intelligible but wooden fails, because
conversion cannot add expression that is not there.

### S7 — End to end in Raaga

The real application path: Raaga generates lyrics and a melody, S6 renders the
guide, conversion applies the selected SPB checkpoint, and the result sits in an
actual song with its accompaniment.

**G7 evidence:** integrated song audio from unseen lyrics, reviewed by Krish.
Integration may be claimed only when the real Raaga path — not a script beside
it — loads the chosen checkpoint and produces that audio. This is the gate that
corresponds to §1, and no earlier gate may be reported as fulfilling it.

---

## 4. How the approach changes: hypotheses, experiments, method changes

This is a research problem with an engineering spine. The spine — provenance,
splits, checkpoints, gates — does not bend. The method does.

**Every experiment is registered before it runs**, in four lines: the hypothesis;
the exact comparison; **what result would count as success and what would count
as failure**, decided in advance; and the cheapest thing that could refute it.
An experiment with no pre-registered failure condition cannot fail, and therefore
teaches nothing.

**Method-change rule.** A method is revised or abandoned when any of these holds:

1. Its pre-registered success criterion fails **twice on distinct evidence** —
   not one inconclusive listen.
2. A cheaper method reaches the same criterion.
3. Its blocking prerequisite is refused or unobtainable — for example, source-use
   clearance, or a Tamil-capable synthesiser.
4. It cannot be evaluated at all: no measurable difference and no listening
   verdict after two attempts to construct one.

A revision is recorded with what was tried, what happened, and what changed —
**failed and inconclusive results stay in the record**. A workflow that keeps
only its successes cannot show whether anything helped.

### The next concrete experiments

**E1 — Tamil singing-synthesis feasibility. Highest information, run first.**

- *Hypothesis:* a singing synthesiser exists, or can be adapted, that renders
  Tamil lyrics at given notes well enough to drive conversion.
- *Comparison:* survey available singing synthesisers for Tamil or adaptable
  phoneme coverage; render **one** short Tamil phrase at fixed notes with the
  most promising; Krish judges pronunciation and naturalness.
- *Success:* recognisable, correctly pronounced Tamil at the intended pitches,
  even if the timbre is wrong — timbre is conversion's job.
- *Failure:* no obtainable model produces intelligible Tamil singing.
- *Why first:* **the entire end-to-end objective in §1 depends on it.** It needs
  no SPB data, no source clearance and no admitted clips, so it can run while
  every other prerequisite is still open. If it fails, the plan changes shape,
  and it is far better to learn that now than after S1–S5 are complete.
- *On failure:* fall back to adapting a multilingual synthesiser with Tamil
  phoneme mapping, or to building a Tamil voicebank from guide-role recordings —
  both larger, both worth knowing about early.

**E2 — Clean guide versus the processed guide. Runs in parallel; identity track.**

- *Hypothesis:* U1 and U3 — the ring originates in the Praat guide processing, so
  a clean guide removes it from conversions.
- *Comparison:* one conversion from a clean guide against preserved 04,
  everything else identical, masked and level-matched.
- *Success:* the ring is gone and SPB character is retained.
- *Failure:* the ring persists, which returns the vocoder and reference
  hypotheses (U7, U8) to scope.
- *Cheapest refutation first:* the pitch-only versus duration-only pair, which is
  signal processing alone and needs no model.

Both are proposed to the audio owner as the execution owner. Neither is started
by this document.

---

## 5. Evidence discipline

- A vocoder roundtrip is a diagnostic **on that input distribution**. It is not a
  universal perceptual ceiling.
- Failed and inconclusive comparisons stay in the record. A workflow that only
  retains successes cannot show whether training helped.
- More training is not guaranteed improvement. The learning curve is observed,
  not assumed.
- No stage may fabricate approval, admit a source, or open its own gate.

**Listening feedback is captured against exact samples.** Every verdict records
the precise file and interval it refers to, its hash, the question asked, and the
measured update it caused — or explicitly that it caused none. A verdict with no
identified sample is not evidence, and a verdict that changes nothing measurable
is recorded as such rather than quietly absorbed.

### Recorded listening evidence

| Date | Item | Krish's verdict | Scope |
| --- | --- | --- | --- |
| Sep 9 | `krish-repeated-phrase-native.wav` (untouched original) | "the original is fine" | That recording only. |
| Sep 9 | `A-guide-then-B-vocoder.wav` | Ring present in A, **lesser** in B | The processed guide rings; the vocoder reduced it on this input. |
| Sep 9 | `untouched-vocoder-control-20260909/B-reconstructed-matched.wav` | "the B reconstructed matched dot wave looks good" | **Positive for that specific reconstruction only.** Not zero-ringing proof, not a universal ceiling, not trained-voice acceptance, and not an SPB identity judgement. |

---

## 6. Uncertainty register

Assumptions currently held that are **not** results. Each names what would
settle it. Several are mine and were stated too strongly earlier today; they are
recorded here rather than left in anyone's memory.

| # | Claim | Status | What would settle it |
| --- | --- | --- | --- |
| U1 | The mechanical ring originates in the Praat guide processing | Supported by three listens; not measured | The pitch-only versus duration-only pair, and a conversion from a clean guide |
| U2 | Nasal codas at syllables 4 and 6 are the ring sites | My reading of the boundary and note arrays; unverified | Spectral inspection of those intervals |
| U3 | A clean guide will remove the ring from conversions | Hypothesis | One conversion on a clean guide against preserved 04 |
| U4 | 5–30 minutes is the useful data band | **Withdrawn as a threshold.** My rule of thumb, not evidence here | Measured learning curve on admitted minutes |
| U5 | One microphone and one room are required | **Withdrawn.** Too strong; would make historical material inadmissible | Conditions labelled; diversity judged by coverage against artifact risk |
| U6 | Missing upper harmonics cause the mechanical quality | **Withdrawn.** Matched H1 figures contradict it — 0.512 source, 0.505 roundtrip, and preference *improved* at a higher share | — |
| U7 | The vocoder is the primary artifact source | **Withdrawn.** The roundtrip reduced the ring | — |
| U8 | Accompaniment in the 8-second reference degrades conversion | Hypothesis; the reference is a song mix | A clean solo reference of comparable duration, all else fixed |
| U9 | Interview speech benefits a singing objective | Unknown | Speech-only and singing-inclusive training compared on the same auditions |
| U10 | Clips longer than 2 s fit the GTX 1060 | Unmeasured. 2 s is the largest tested | Measured memory at increasing durations |
| U11 | An hour of training is stable, and a large checkpoint saves | Unmeasured | The S4 pilot |
| U12 | Raaga can compose in a chosen key or register | Unverified | Inspection of the composer path |
| U13 | A singing synthesiser with adequate **Tamil** coverage is obtainable | Unresolved, and the main risk to S6 | Survey of available models and voicebanks; a rendered Tamil phrase judged for pronunciation |
| U14 | A synthetic guide can carry enough expression for conversion to inherit | Unknown | S6 audition: same lyrics from a synthetic guide and a recorded guide, converted identically |
| U15 | Fine-tuning the conversion model improves pronunciation, pitch and timing, not only timbre | Krish's expectation; plausible for timbre and vowel quality, unestablished for the rest | Checkpoint auditions judged separately on words, melody, expression and naturalness |
| U16 | The conversion step remains necessary once S6 exists | Assumed. If a synthesiser could sing in SPB's voice directly, it would not be | Compare S6-then-conversion against any direct path that becomes available |

---

## 7. Current state

| Artifact | Revision | Status |
| --- | --- | --- |
| Preflight | `831e045` | Accepted, reused unchanged |
| Runner | `f7894c0` | **Changes required** — S1 |
| GPU step review | `5ece4fc` | Verified |
| Recording session draft | `64a101f` | Delivered; guide-versus-target distinction stands |
| Asset contract, pending plan | owner's | Ten assets verified; plan refuses on data |
| Admitted SPB clips | — | **Zero.** Source use unresolved |
| Score-to-performance component | — | **Does not exist.** S6; the installed path has no text or phoneme input |
| Lyrics, fitting, note and duration targets | in Raaga | Exist and are reused by S6 |

**Open prerequisites, none of which this document resolves:** SBS source-use
clearance; availability of clean solo SPB *singing* rather than interview speech;
whether a clean guide removes the ring; the compute measurements in U10 and U11;
and whether a Tamil-capable singing synthesiser is obtainable for S6.

**Interim dependence, to be labelled on every report until S6 delivers:** every
conversion result currently rests on a **recorded guide**. Such a result speaks
to identity and audio quality. It is not evidence that new lyrics can be sung,
and must not be reported as progress toward §1's end-to-end outcome.
