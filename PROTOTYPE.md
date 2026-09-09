# Voice prototype — implementation note

Built to `docs/spec/VOICE_PROTOTYPE_CONTRACT.md` at **`46e5ca2` (v2)**, which is
the base of this branch. Isolated: no production Raaga code is touched, and
incorporation follows a good result, independent review and Krish's listening
judgment.

**Tamil lyrics + a tune + a target voice sample → one sung audio file.**

## The shape

```
lyrics (Tamil)   ─┐
                  ├─ frontend ──► alias requests, or a refusal by name
slots (aksharas) ─┘
tune (events) ────► continuous pitch curve + voicing mask (rests stay rests)
                          │
       requests + placement ──► carrier ──► guide.wav
                                              │
                          reference (hash-checked) ──► converted.wav
                                              │
                                     listening copies, gain only
```

Every stage writes into one render directory, and the record names all five
things by hash: reference, tune, lyrics, output, feedback.

## The decisions worth knowing

**Direct melody controls, not extracted pitch.** The tune arrives as events with
fractional MIDI and explicit bends. Re-extracting f0 from audio something already
knows symbolically is a lossy round trip we would be choosing. A monophonic WAV
adapter exists for the case where only audio is available; it marks its output as
an estimate, and its f0 is autocorrelation cross-checked against a harmonic sum,
because both of those methods were wrong by an octave in opposite directions on
real renders during this work.

**Real samples driven by a continuous curve.** The carrier plays a CVVC bank of
recorded vocal samples, and drives them from the tune's own pitch curve rather
than one MIDI number per note. The samples carry vowel and consonant identity
that a formant model approximates; the curve carries the ornamentation that a
note grid destroys. Neither approach on its own gets both.

**Phrases are synthesised separately and placed.** That is what makes a rest
silent in the audio rather than only in the record.

## What it refuses

| Situation | What happens |
| --- | --- |
| Tamil the bank cannot say | Refused by name, with the alias and the reason |
| Empty lyric | Refused — never a successful silence |
| A lyric too dense to be sung | Refused, rather than compressed until unsayable |
| A syllable that would pass the end of the tune | Refused |
| A reference whose hash has changed | Refused before the run |
| A render directory or file that exists | Refused — nothing is overwritten |
| Feedback with no author, no text, or an invented disposition | Refused |

## Trial 1

`python -m trials.nilavaai_two_lines [--convert]`

**நிலவாய் நீ வந்தால் இரவும் பகலாகும்** — *if you came as the moon, the night
would become day*. Authored for this experiment; not from any released song.
Thirteen aksharas over two lines with a **0.9 s rest between them**, 12.50 s
total — inside the contract's 10–20 s range, and deliberately containing the
thing the reviewed implementation sang through.

Guide stage measured: 13 syllables placed, 2 phrases, ends at 12.50 s against a
12.50 s tune, peak 0.890, **0 samples at the rail**, and **rest peak 0.000000
across 90 rest hops**.

Without `--convert` the record carries a warning that a carrier alone does not
complete the milestone, because it does not.

## Tests

```
python -B -m unittest discover -s tests -t .
```

32 pass. Fixture-only except the two carrier tests, which use the installed bank
and engine on the CPU — no model, GPU, network, corpus file or training. The
reviewed defects are tested first: Tamil to silence, alignment overflow, rests
sung through, clipping reported as clean, renders overwritten, and feedback
outliving the bytes it described.

## Explicitly unavailable

Corpus construction and training are **not implemented here and not stubbed**.
Separation is unavailable, so no mixed audio can be labelled isolated; dataset
rebuilds and splits belong to `SPB_VOICE_TRAINING_WORKFLOW.md`. This milestone is
**inference only** — no weights are updated, and no comment or control change is
ever presented as training.

## Dependency held outside this branch

`tamil_frontend.py` is imported by path from `Claude-Tamil-Frontend-20260909`
rather than copied, so the Tamil rules have one home. Its hash goes into every
render record. That workspace's `237a87f` listening package is untouched.
