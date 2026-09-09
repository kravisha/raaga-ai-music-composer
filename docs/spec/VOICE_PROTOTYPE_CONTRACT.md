# Voice prototype — implementation contract

**Revision history.** v1 is `5ee63d7`, preserved unchanged in history. This is
**v2**, which adds §1.5 (clip scope) and amends §1.1; nothing in v1 is removed or
reversed. Cite the revision you are working from.

The requirements both developers start from for the authorized isolated
prototype: **target voice sample + Tamil lyrics + tune → a real sung audio
file**, exposed for listening, with feedback recorded durably and revised
renders attributable.

Companion to `SPB_VOICE_TRAINING_WORKFLOW.md` (spec `ec41179`). That document
covers sources, clips, training and evaluation. This one covers the prototype
that produces audio *now*, from an existing authorized reference and existing
inference — **no training run, and no production integration.** Incorporation
into Raaga follows a good result, independent review and Krish's listening
judgment, in that order.

---

## 1. Inputs

### 1.1 Target voice sample

Recorded before use, and **verified before every render**:

| Field | Rule |
| --- | --- |
| `path` | Where it lies. Read-only; never copied into the prototype. |
| `sha256` | Checked against the record **before** the run. A different file under the same name is a different voice, and the run stops. |
| `use_scope` | Carried verbatim from the record that authorized it. |
| `limits` | Every recorded limitation travels with every render made from it. |

For the first trial: the existing SPB-credited reference, already used for
authorized local inference, with its recorded limits — a song mix with
accompaniment rather than an isolated vocal stem, one 8-second excerpt of one
song, local private trial scope.

**Its limits are carried, never edited away.** A song mix is never relabelled as
an isolated vocal because a later stage would prefer one, and an existing
reference is **not padded** to reach a nominal duration. Eight seconds that are
real beat twenty that are partly invented.

**Inference conditioning is not corpus admission.** Using an authorized
reference at inference time confers nothing on the training source store, and
admission there remains a separate decision.

### 1.2 Lyrics

- **Script:** Tamil, as written. Not a romanisation.
- **Transliteration convention:** stated explicitly whenever Latin text is
  accepted at all. A Latin vowel spelling is meaningless without one.
- **Vowel length is contrastive and must survive.** `padi` and `paadi` are
  different words. So are the vowel identities: `ee` is not `i`, `oo` is not `u`.
- **Consonant grouping must survive.** `dh`, `th`, `ch`, `sh`, `zh`, `ng`, `ny`
  are single consonants, not a consonant plus `h`.
- Anything the selected backend cannot say is **refused by name, with a reason**.
  Never approximated silently: a substituted neighbour sounds plausible and says
  a different word.

### 1.3 Tune

**Prefer direct melody controls.** Where the melody is known symbolically —
pitch, duration, voicing per event — take it that way. Re-extracting f0 from
audio the engine already knows symbolically is a lossy round trip we would be
choosing.

**A monophonic-WAV adapter is documented and permitted** where only audio
exists. When used, the record says so, names the extractor, and states that the
curve is an estimate.

Either way the tune is represented as:

- a **continuous pitch curve**, never quantised to semitones — quantising is
  what destroys gamaka, meend and microtonal placement;
- an explicit **voicing mask**, where a rest is unvoiced and *stays* unvoiced;
- note events used **only** to place syllables, never to replace the curve.

### 1.4 Refusal, not successful silence

Empty, unsupported or unparsable content **fails with a reason**. A run that
returns success, writes an audio file and contains none of the requested words
is the worst available outcome, because nothing downstream can detect it.

### 1.5 Clip scope — short first (v2)

Iterations are **short and complete**: roughly **10–20 seconds**, beginning with
one short complete phrase. A complete short phrase is the unit, not an arbitrary
window, and a shorter fixture that is already useful is not padded to reach the
range.

**Full recordings remain valid sources and are retained.** Short-clip scope is
about what is fed to the prototype, not about discarding anything: whole
recordings stay where they are, unmodified, and bounded excerpts are taken from
them. Short-clip scope **never** rejects, deletes or supersedes a full source
recording, and as results warrant the curated sample set is *expanded* from those
same sources under the applicable workflow — which is why the parentage below is
recorded at extraction time rather than reconstructed later.

Every clip records its parentage, so expansion later keeps provenance rather than
reconstructing it:

| Field | Rule |
| --- | --- |
| `parent_source` | The full recording it came from, by path and hash. |
| `interval` | Start and end in the parent, in seconds. |
| `operation` | What was done — excerpt and channel handling only, no processing. |
| `limits` | The parent's recorded limitations, carried to the clip. |

**A whole video is never fed to the prototype.** Where only a long source exists,
a bounded excerpt is taken and recorded as above.

**One 10–20 second example is not a training set.** If weight training is later
introduced under `SPB_VOICE_TRAINING_WORKFLOW.md`, it uses a *curated collection*
of short clips with the split and leakage rules of §S3. No single short example
may be described as sufficient for training, and the current milestone remains
**inference, not training**.

### 1.6 One iteration, linked (v2)

Each iteration links its five things to each other by identity, not by being in
the same folder: **reference, tune, lyrics, output, feedback**. Every render
record names all five with their hashes, and every feedback entry names the
render and the audio bytes it was given.

---

## 2. Outputs

### 2.1 Versioned, never overwritten

Every render gets its own directory under a render id. **A previous render is
never overwritten or replaced**, and a destination that exists is a refusal.

Each render directory holds, at minimum:

- the resolved **inputs** — lyric text, tune controls, reference identity;
- the **guide** (carrier) audio, as produced;
- the **target-conditioned** audio, as produced — the raw, unmodified;
- a **listening copy** at a stated level, gain only;
- a **render record** (below).

### 2.2 The render record

| Field | Rule |
| --- | --- |
| `render_id`, `parent_render_id` | What this is, and what it was revised from. |
| inputs + hashes | Lyric, tune, reference, and the hash of each. |
| `backend`, model, config, checkpoint identities and hashes | What actually ran. |
| seed and every setting | Including which are held constant against a named comparison and **which are not**. |
| timing, device, precision | Measured, not intended. |
| `peak`, `samples_at_rail` | **Measured.** A tolerance that admits full scale is not a clipping check. |
| processing | Exactly what was done to the audio, including "gain only". |
| failures | Recorded, never swallowed. |
| `listening_status` | Defaults to *not heard*. |

**Honest level reporting.** If samples reach the rail, say so and say how many.
A quieter copy of a clipped file is quieter, not repaired — only a re-render
with more headroom repairs it.

**No naturalness or identity acceptance may be asserted by any tool.** Output is
*synthetic reference-conditioned audio*, clearly labelled, and never described as
an authentic performance by the target singer.

---

## 3. Feedback

Durable, and bound to exactly what was heard.

| Field | Rule |
| --- | --- |
| `render_id` | The exact render. A verdict on one render is not a verdict on another. |
| `render_sha256` | The bytes heard. If they differ now, the feedback is stale and says so. |
| `author`, `timestamp` | Who, and when. |
| `text` | What they said, verbatim. |
| `segment` | The part it refers to, where it refers to a part. |
| `disposition` | One of *preferred*, *not preferred*, *no preference*, *needs work* — with **no preference an answer**, not a missing one. |

**Viewing a result does not approve it.** Absence of feedback is absence of
feedback, and defaults to *not heard*.

**A comment is not a weight.** Feedback records what a person said. It does not
update a model, and nothing may present it as having done so.

---

## 4. Render, controls, training — three different things

| Kind | What changed | Evidence required |
| --- | --- | --- |
| **New render** | Nothing but the run | Parent render id, identical control record |
| **Changed controls** | Settings, reference, lyric or tune | The specific fields, before and after |
| **Training** | Model weights | Its own dataset, checkpoint and run evidence, under `SPB_VOICE_TRAINING_WORKFLOW.md` |

The first milestone uses **existing inference only** and requires no training
run. A parameter change is never labelled as learned weights.

---

## 5. Defects this contract exists to prevent

Each reproduced in review, and each is a rule above rather than a warning.

| Defect | Rule |
| --- | --- |
| Tamil script stripped to zero syllables, silence returned as success | §1.2, §1.4 |
| Vowel length and identity collapsed | §1.2 |
| Consonant digraphs split | §1.2 |
| Syllables overflowing past the end of the tune | §1.3, §6 |
| Rests sung through with nominal pitch | §1.3, §6 |
| Missing separation labelling mixed audio as isolated | §7 |
| Stale excluded clips surviving a dataset rebuild | §7 |
| A clipped render reported as clean | §2.2 |
| Outputs silently overwritten | §2.1 |

## 6. Invariants

1. The pitch curve is never quantised to semitones.
2. **A rest is silent in the output.** Keeping an unquantised array is not
   enough if a later stage sings through the silence.
3. **No syllable extends past the end of the declared tune.**
4. Every stage writes a record with the hashes of what it read and wrote.
5. Nothing unsupported is approximated; it is refused by name.
6. No render is overwritten.
7. Nothing outside the prototype's own workspace is written — caches, working
   directories and outputs included.

## 7. Explicitly unavailable in this milestone

Corpus construction and training paths are **not part of this prototype and are
not to be stubbed as if ready**:

- Automatic separation is unavailable. Mixed audio is **never** labelled isolated
  vocals; a missing separator fails closed or requires explicitly verified clean
  input.
- Dataset rebuilds must reflect the current manifest exactly, with collision-safe
  provenance and the family split rules of `SPB_VOICE_TRAINING_WORKFLOW.md` §S3.

Where these are not implemented, they are reported as **unavailable**, never as
passing.

## 8. Ownership

Claude codes and tests. Codex/Arya reviews independently. Krish judges the
singing, and his listening is separate from technical acceptance. Audio and
compute runs are coordinated through the existing relay before execution; fixture
coding and tests proceed independently.
