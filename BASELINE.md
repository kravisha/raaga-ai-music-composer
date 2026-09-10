# Active baseline — morning 04. Synthetic-guide exploration paused.

Krish's final listening decision, 2026-09-09:

> E is bad, not acceptable. I don't think this path is acceptable at this time.
> Let's go back to where we were at the seventy percent morning. Regroup from
> there.

**E is rejected. This exploration is paused. Morning 04 is the active quality
baseline.** The ~70% is his subjective rating — not a measured score, and not
production acceptance.

## The baseline

| | |
| --- | --- |
| Original | `Raaga-Vocal-Quality-20260909/spb-core-resonance-20260909/audio/04-reduced-core-guidance.wav` |
| Reproduced | `renders/morning-control/04-reproduced-reduced-core-guidance.wav` |
| **Both SHA256** | **`1e9da45d7daf5274fbaec7de0bb9b9ed326f021a1593f905eef468c68d3c5f97`** |

Byte-identical. All 199,168 samples at 44100 Hz, 4.516281 s. The raw containers
differ only in the WAV `PEAK` chunk's write timestamp.

**What it is:** a synthetic SPB-reference-conditioned conversion of a recorded
performance of «திரும்பி வந்தாய் அன்பே», word-timed and lowered in register.
**Not** newly synthesised current lyrics, and **not** an authentic SPB recording.

**Correction, and it is mine.** I repeatedly called that input guide "Krish's own
recorded voice". **The evidence does not support that**, and I should not have
said it. The attribution came from a filename — `krish-repeated-phrase-native.wav`
— and from a hardcoded string in an older script. A filename is not evidence of
who performed, which is the exact standard I have applied to everyone else's work
all day and failed to apply to my own. The chain traces to `WhatsApp Audio
2026-09-08 at 7.00.14 PM.mp4`; **the performer of that guide is not established**
and is being checked separately. Krish identifies the audible voice in morning 04
as SPB, not himself — and that is the *output* of the conversion, which is a
different thing again from whoever sang the input.

Three distinct things, kept apart from here on: the **input guide** (performer
unestablished), the **conditioning reference** (the SPB-credited song excerpt),
and the **output** (SPB-conditioned synthesis, not an authentic recording).

Input `05-recorded-voice-word-timing-lower-register.wav` `e650a658…`; reference
`spb-credited-reference-8s.wav` `7ba960f2…`, an accompanied song excerpt rather
than an isolated vocal stem. Seed-VC `51383efd…`, DiT `42aef93f…`, config
`fbca8cca…`, CFG 0.3, 30 steps, three seeds 20260909, CPU float32, 4/1 threads,
length 1, F0 on, auto F0 off, shift 0, raw FLOAT; listening applies the preserved
−0.779490993561403 dB scalar at PCM16. **No training.**

## What is preserved, in place

C `r20260910T003437Z-01`, D `r20260910T005521Z-01`, E
`current-lyrics-morning-settings` — every raw, guide, listening copy, setting and
record stays exactly where it is. **No deletion, move, reset, revert or
production-profile change** is part of this decision. They are experiment
history, not baselines.

E's rejection is recorded in `feedback.json` bound to the bytes actually heard,
`f1959b14…`, alongside every earlier verdict.

**E's technical checks all passed** — exact source, all settings, no raw
excursions, no listening rails, scalar-only level matching within one quantum.
**None of that established acceptable singing.** That is the lesson worth keeping
from this stage: the measurements were right and the result was still rejected.

## Hold

No further current-lyrics or synthetic-guide render, no isolated tail, no
transposition, no training, no other render. **No automatic request for a new
recording.** The guard cleanup at `6a8bddd` is complete and independently
verified on both branches; it is not to be extended.

Next action is to **regroup from morning 04** — a natural recorded performance
with its known shortcomings, kept intact for that discussion.
