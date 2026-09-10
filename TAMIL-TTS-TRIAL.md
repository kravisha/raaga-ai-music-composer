# Tamil TTS trial — stages 1–2 delivered, guide gate failed, stopped

One trial. **No SeedVC conversion was run**, no sweep, no training, no second
TTS. Everything below is in `renders/tamil-tts-trial/`.

## Model

| | |
| --- | --- |
| Model | `facebook/mms-tts-tam` |
| Revision | `e9cf59dae34f0f51e3b1842876a658e4516f9fe4` |
| Files | `config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json` — **safetensors only**, no `pytorch_model.bin`, no remote code |
| Weights | 145,242,872 bytes, `29357e85c7f86f77…` |
| Total | 145.2 MB |
| Licence | **CC-BY-NC-4.0** — isolated private trial, no distribution, no production enablement |

Sources, sizes and per-file hashes are in `models/facebook--mms-tts-tam/acquisition.json`.
Nothing else was downloaded and no package was added or upgraded.

## Stage 1 — speech

All six words, exactly as given: **நிலவாய் நீ வந்தால், இரவும் பகல் ஆகும்**.
`பகல்` and `ஆகும்` are separate; nothing was substituted. The comma was removed
for synthesis and the original is recorded beside it.

`01-tts-raw.wav` — 4.128 s, 16 kHz, peak 0.700, no excursions,
`5fab9d9ccf2612f3…`.

Durations come from the model's own duration predictor, captured off the module
and scaled by `1/speaking_rate` exactly as its length regulator does. The 73
token timings **sum to 4.128 s**, matching the waveform — which is what makes
them the model's timings rather than my arithmetic. They are **character**
timings from a tokenizer with `phonemize=false`; not validated phonemes.

Unprompted ASR: **3 of 6** verbatim —
`நிலவாய் நீ வந்தால் ஏறவும் பகல்லாகும்`.

## Stage 2 — shaping, and the bug found in it

Score: the morning packet's **seven notes**, morning lower-register MIDI
62/51/54/55/59/56/55, durations .565/.2825/.2825/.565/.565/1.13/1.13. Not the
13-event `nilavaai` fixture; not inferred from audio.

**A real pitch bug, found in review.** The first retimer advanced its read
pointer a whole period and its write pointer `period − fade`, so repeats landed
out of phase and the overlap moved the pitch — 200 Hz came back at 266.67 Hz,
**+498 cents**. `trials/periodic_retime.py` replaces it with ordinary PSOLA:
two-period Hann grains, output hop exactly one period, read positions snapped to
whole periods, exact length by allocation. Invariant verified — 200 Hz →
**199.951 Hz, −0.42 cents**, exact length, confirmed independently in both
directions.

`07-shaped-guide-pitchfixed.wav` — 4.52 s, realised total 4.519999 s, peak
0.446, no rails, `5f109ca6…`. The failed `05` guide is kept unchanged.

## Both gates, stated with their limits

**F0 — conditional pass.** Median **1.05 cents** from target, max 20.3 — but
over **10 of 13** syllables only. The first three (`நி`, `ல`, `வாய்`) returned
**no pitch** at their measurement points, so **the opening notes are not
validated**. "The notes are in tune" is true of ten of them and unproven of
three.

**Words — fail. 0 of 6.** Transcript
`நான் நான் விட்டும் நான் விட்டும் விட்டும்` — repetitive, which is the
signature of repeated vowel material rather than a held note.

**Onset and coda bounds are voiced-region estimates, not acoustic phoneme
proof.** Nothing here shows a nasal or voiced coda survived.

## What actually went wrong — and a correction to my own conclusion

I wrote that the score must match the lyric, presenting 13 syllables on 7 notes
as the cause. **That was too quick.** The measurements show a plainer fault, and
it is mine:

| syllable | target | kept as onset+coda | left for the vowel |
| --- | ---: | ---: | ---: |
| `வாய்` | 0.2825 s | **0.2650 s** | **0.0175 s** (280 samples) |
| `ல` | 0.0499 s | 0.0031 s | 0.0468 s |
| `கும்` | 0.678 s | 0.3439 s | 0.334 s |

My partition extended every syllable span to the **next syllable's start**, so
the silence between words was absorbed into the span — and then "protected" as
coda. `வாய்` kept a quarter-second of mostly inter-word gap as consonant and
left the vowel 17 milliseconds. A vowel that short, tiled out to fill a note,
is exactly what produces a repeated syllable.

**So this is an alignment and retiming fault to investigate, not proof that
seven notes cannot carry thirteen syllables, and not evidence that the TTS
approach fails.** The extreme ratios are real (`இ` at 5.89×) and may still
matter, but they are not established as the cause while a bug of this size sits
in front of them.

## Final state

| | |
| --- | --- |
| Commit | **`8c7d7d66e9f490798c2096b6d21763355c127930`** |
| Branch | `claude/voice-prototype` |
| Contract | `46e5ca2` |
| Jobs running | **none** |

Preserved unchanged: raw TTS `01`, failed guide `05`, corrected guide `07`, the
SPB reference, morning 04 original and reproduced, and C, D, E.

Stopped for listening. No further shaping, model or conversion.
