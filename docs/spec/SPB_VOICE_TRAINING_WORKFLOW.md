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

### The capability gap, corrected

An earlier revision of this document said the system has no score-to-performance
component and that every conversion needs a **recorded** guide. **That was
wrong**, and the correction matters because it changes what S6 is for.

What already exists, verified:

- `raagacomposer/voice/renderer.py` — `plan_segments` (line 103) and
  `render_melody` (line 346) render from melody plus lyrics with a
  source-filter/formant synthesiser. Limited in quality; present.
- `Raaga-Tamil-CVVC-Trial-20260908/` — `render_trial.py` and
  `render_expressive_trial.py` rendered the authored Tamil phrase at explicit
  notes through the installed Hanakoloid bank and Worldline renderer. Aliases
  were supplied by hand, so this is not a complete automatic frontend. Version
  one was rejected by Krish for mechanical sound and poor quality; version two
  is not listening-accepted. Pitch coverage within 50 cents was measured at
  100/86/100/100/98/100/100 per note — which supports the target pitches and
  says nothing about naturalness.
- `Raaga-Personal-Voice-Trial-20260908/prepare_local_trial.py:48` copies the
  **synthetic** expressive CVVC output to `inputs/source-generic-v2.wav`, and
  `run_conversion.py` converts from it. `output/run-status.json` records
  `rendered_not_accepted`. **Lyrics → synthesised guide → conversion has already
  been run end to end.** That historical run used a Krish identity reference and
  is capability evidence only; it is not permission to substitute a personal
  target now.

**The actual gap** is therefore not the absence of synthesis. It is the absence
of an **accepted, integrated, natural Tamil lyric-and-score-to-performance path**
with automatic pronunciation and alignment and expressive delivery. The existing
renders failed on delivery, not on existence, and the frontend is manual.

**A guide is a performance waveform, human or synthesised**, distinct from the
target and from the reference. Recorded-guide dependence is therefore not a
property of the architecture; it is a property of whichever guide a given run
used, and **each result records which kind it used**. A result driven by a
recorded guide is an identity and quality result and is not evidence that new
lyrics can be sung.

A directly trained singer — lyrics and notes to SPB audio in one model — remains
an option throughout. Choosing it would not require discarding the conversion
baseline or the comparison infrastructure.

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

**Two tracks.** S1–S5 are the **identity** track: they make a converted voice
sound like SPB and sound natural. S6 is the **performance** track: it makes a
sung guide from lyrics and notes. Neither track alone fulfils §1. They meet
at S7.

### Dependency edges, stated as edges rather than as an order

An earlier revision said "gates open in order" while also requiring E1 before
S1–S5 finished, and required application integration at G5 although integration
is S7. Those were circular. The actual constraints:

- **May run now**, needing no gate: original and control diagnostics; S6
  inference experiments; conversion from the **pretrained base** identity path
  for exploratory S6 comparison. A pretrained identity path does not wait for
  new SPB fine-tuning.
- **G1, G2 and G3 precede real S4 training.** Corrected runner, admitted
  sources, verified splits.
- **The pretrained-base audition baseline is captured before the first update.**
- **Pilot evidence precedes the extended run** (G4).
- **S5 selects and evaluates the trained candidate.** It does **not** require
  application integration.
- **S7 integrates the independently accepted identity and performance paths**
  and owns actual-song acceptance.

One execution owner per stage, non-overlapping claims. Parallel work is expected
where claims do not overlap.

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

**Three outcomes, kept distinct, and represented in the schema.** A source is
`admitted`, `review_pending` (plausibly useful, with the specific unresolved
items named in `pending_on`), or `unusable` (with the reason recorded).
**Pending is not unusable**, and collapsing the middle case in either direction
is how good material gets discarded and unreviewed material gets used.

**Available input, already assessed by the corpus owner** and reused rather than
rediscovered: `local-corpus-inventory-20260909/inventory.json` over the known
roots — 599 MP4 and 307 WAV files, **36 filename-selected SPB-related media in
33 candidate upload families, approximately 155 raw minutes** after grouping
same upload IDs, all with readable audio metadata. Two are single-credit
priority candidates; many are co-credited songs; two are tribute or cover
candidates. **This is potential material with review work outstanding, not clean
or admitted duration.** Same-ID variants and likely alternate uploads stay in
one split family. The interview inventory remains separate. Originals unchanged.

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

  // The outcome is stated, not inferred from a boolean. Technical utility and
  // source-use status are separate axes: material can be technically excellent
  // and unusable on rights, or cleared and technically unusable.
  "outcome": "admitted | review_pending | unusable",
  "outcome_reason": "why, in words",
  "pending_on": ["the specific unresolved items, when review_pending"],
  "technical_utility": "usable | marginal | unusable | unassessed",
  "decided_by": null, "decided_utc": null
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

**Stop rules, with their actual values fixed before the run and recorded in the
`run_manifest`.** Declaring that a rule exists is not the same as declaring its
value, and only the value is executable. Each run records:

| Rule | Recorded value |
| --- | --- |
| Checkpoint cadence | every N steps, N stated |
| Audition schedule | which checkpoints are auditioned, and when |
| Lack-of-improvement rule | how many consecutive auditions without preference end the run |
| Wall-clock budget | seconds, inclusive of checkpoint and evaluation overhead |
| Resource ceiling | the memory fraction and what exceeding it does |

These are per-run values, not universal constants. Also stopping the run: user
cancellation, a non-finite update, a failed checkpoint commitment, and resource
failure. Keep an evaluated best checkpoint and a recovery checkpoint. **Never
publish the latest merely because loss fell.**

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
| Stable, natural phrasing | Conversion inherits prosody, so a stiff guide is expected to give a stiff result. Both existing CVVC renders were rejected for mechanical delivery | The standing failure to beat |
| Declared processing, retained originals | Every processing step named, originals kept, settings fixed, repeatability measured | Required |
| Local and offline by default | The present default, not a prohibition. A method that needs otherwise is proposed with its justification | Default |

**Data requirements.** A voicebank or model that sings Tamil. Whether this is
sourced, adapted, or built from recordings is open. If it is built, its training
data is **guide-role** material governed by the same admission rules — and still
not SPB target data. A selected bank or model needs its **own applicable
use basis**, which is a different question from SPB source clearance and is not
satisfied by it.

**No method is prohibited by an unproved cause.** The ring's origin is a
hypothesis (U1), so it may not be used to forbid a processing category. What is
required is that processing be declared, originals retained, settings fixed and
repeatability measured.

**G6 evidence:** an **unseen lyric sequence** rendered to a guide and then
converted, in a masked level-matched audition. Krish judges Tamil pronunciation,
melody and timing, expression, and naturalness separately. A guide that is
intelligible but wooden fails, because conversion cannot add expression that is
not there.

**"Unseen" is defined relative to this project, because it cannot be defined
relative to a pretrained model.** An unseen lyric sequence is one that appears in
none of the project's adaptation data, tuning material, references, or the
material used to select a checkpoint. Individual words and phonemes will
certainly have been seen; the *sequence and performance* must not have been. Each
audition records the check that was made, and states plainly that **overlap with
a pretrained model's own corpus is unknown**. A final reserved source-family test
stays separate from validation material used repeatedly.

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

**E1 — Beat the existing Tamil render on novel phrase material.**

Not a rediscovery of whether Tamil synthesis is possible: two renders exist and
the path to conversion has already been run. The question is whether delivery
can be made acceptable.

- *Baseline:* the existing `tamil-cvvc-expressive-v2.wav` and its recorded
  rejection, plus version one's, reused rather than re-derived.
- *Hypothesis:* a bounded improvement to the existing renderer, or one new
  candidate synthesiser, produces a materially less mechanical Tamil
  performance on a **novel phrase** at explicit notes.
- *Comparison:* the existing render and the new one, same phrase, matched level,
  masked order. Krish judges pronunciation, melody and timing, expression and
  naturalness separately.
- *Success:* the new render is preferred on delivery while keeping pitch
  agreement at least comparable to the measured baseline.
- *Failure:* it is not preferred, or intelligibility drops.
- *Reporting rule:* a result names **which candidate, which phrase, which
  settings, and how wide the search was**. One render that disappoints is
  **inconclusive about that candidate**, not evidence that no obtainable model
  works. Two failures trigger a method review under §4; they do not establish
  impossibility and do not justify discarding the evidence.
- *Prerequisite:* a selected bank or model needs its own applicable use basis.
  That is not SPB source clearance and is not a generic approval for routine
  inspection.
- *Why early:* it needs no SPB training admission and no admitted clips, so it
  runs while those remain open.

**E2 — Guide comparison. Runs in parallel; identity track.**

- *Hypothesis:* U1 and U3 — the ring originates in the guide processing.
- *Honest framing:* an unprocessed guide and processed 04 differ in pitch contour
  and durations as well as in processing. **This is a bundled-input comparison
  and cannot isolate the cause**, unless the performed score and timing are held
  within explicitly stated tolerances. It is labelled as such either way.
- *The control that can isolate it:* a **pitch-only versus duration-only
  factorial** from the same original — signal processing alone, no model. Even
  so, a failure there does not rule out other causes and does not establish a
  vocoder or reference cause.
- *Success:* a preferred guide is identified, with the comparison's bundled or
  factorial nature stated.
- *No causal claim from phoneme boundaries.* The syllable-to-note mapping is a
  reading of arrays (U2), not a measurement, and may not be used to attribute
  cause.

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
| U13 | Tamil singing synthesis at explicit notes is **possible** | **Settled: it is.** Two renders exist. What is unresolved is acceptable delivery and an automatic frontend | E1, against the existing rejected baselines |
| U17 | A recorded guide is required for conversion | **Withdrawn.** A synthesised guide has already driven a completed conversion | — |
| U18 | The syllable-to-note mapping identifies where the ring is added | My reading of arrays, not a measurement, and not usable as a causal claim | Spectral inspection |
| U19 | Overlap between an unseen lyric sequence and a pretrained model's own corpus | Unknown and not verifiable for a pretrained model | Stated as unknown on every audition |
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
| Formant renderer | `raagacomposer/voice/renderer.py` | Exists; limited quality |
| Tamil CVVC renders | `Raaga-Tamil-CVVC-Trial-20260908` | Two renders, **both rejected on delivery**; manual aliases, no automatic frontend |
| Synthetic guide into conversion | `Raaga-Personal-Voice-Trial-20260908` | Completed end to end; `rendered_not_accepted` |
| Accepted natural Tamil score-to-performance | — | **This is the gap.** S6 |
| Lyrics, fitting, note and duration targets | in Raaga | Exist and are reused by S6 |

**Open prerequisites, none of which this document resolves:** SBS source-use
clearance; availability of clean solo SPB *singing* rather than interview speech;
whether a clean guide removes the ring; the compute measurements in U10 and U11;
and whether the existing Tamil synthesis can be improved to an acceptable
delivery, with an automatic pronunciation and alignment frontend.

**Every result records which kind of guide drove it** - recorded or
synthesised. A recorded-guide result speaks to identity and audio quality and is
not evidence that new lyrics can be sung; it must not be reported as progress
toward §1's end-to-end outcome.
