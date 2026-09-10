# Headroom pass — completion record

One authorized control change, one conversion, executed and released. Contract
`46e5ca2` (v2). Render `r20260910T003437Z-01`, parent `r20260910T001153Z-01`.

## What ran

```
.../Raaga-Singing-Experiment-20260908/.venv/Scripts/python.exe -B -m trials.headroom_pass
```

Launch `2026-09-10T00:34:36Z`, exit `2026-09-10T00:38:43Z`. 30 diffusion steps,
RTF 17.34, CPU, float32, offline, pinned local models.

**PID: not captured, and I will not supply one after the fact.** The process was
run in the foreground and had already exited before its id was read. The command,
window and single-renderer ownership are recorded above; the id itself is a gap
in this record rather than something to reconstruct. Future runs capture it at
launch.

**Released.** Nothing of this pass is running. The audio task's pending
coordination call was left intact throughout — not bypassed, not retried.

## The one change

| | peak | rails +/− | RMS |
| --- | --- | --- | --- |
| Baseline guide | 0.8900001049 | 0 / 0 | −13.8053 dBFS |
| Attenuated ×0.5 | 0.4450000525 | 0 / 0 | −19.8259 dBFS |

Ratio exactly 0.500000 within PCM24 rounding; 551250 frames both sides; 44100 Hz;
rests and waveform shape preserved. All thirteen conversion settings and the
reference, checkpoint and config hashes were verified equal to the baseline
record **before** launch, and the run refuses to start if any differ.

## Result

| | peak | rails + | rails − | total | RMS |
| --- | --- | --- | --- | --- | --- |
| Baseline B | 1.0000000000 | 145 | 29 | 174 | −8.27 dBFS |
| New C | 1.0000000000 | 87 | 30 | 117 | −8.6175 dBFS |

Rails counted at the true 24-bit extremes: positive `1 − 2⁻²³`, negative exactly
`−1.0`. The looser 0.999999 count is kept beside them as a diagnostic, not in
place of them.

## Correction to my own conclusion

I reported this as *"the conversion drives to full scale regardless of what it is
given, so this is not an input-headroom problem and no input attenuation will fix
it."* That is an overstatement and the peer review was right to reject it.

**One tested reduction cannot establish "no amount" or "never".** I generalised
from a single point to a universal claim — the exact move I have spent this work
refusing when other people's documents make it.

What the evidence supports: a 6.02 dB input reduction reduced rails from 174 to
117 and left the output peak at exactly 1.0 with RMS within 0.35 dB. **That one
reduction did not eliminate clipping.** It does not show that no reduction would,
and it does not identify a cause.

Open, untested hypotheses: internal normalisation in the conversion; output
limiting in the conversion or vocoder; overflow at the PCM save after inference;
a larger reduction than the one tested; something else. Inspecting the pre-save
float peak and the save path would distinguish several of these **without a model
run** — noted as available, not requested, and not done here.

The render, its record and its notes are unchanged. The correction also sits
beside the render as `conclusion-correction.json`, and the render's own warning
was already correctly hedged: *"a lower input does not guarantee fewer clipped
samples."*

## Also corrected

An equal-note-duration hypothesis about this fixture is **not true of it**. Its
13 sung notes run 0.50–1.90 s with 9 distinct values: 0.8, 0.5, 1.1, 0.9, 0.6,
1.5, 0.7, 0.5, 1.0, 0.7, 0.6, 0.8, 1.9 — confirmed from the trial's own input
rather than from the audio.

## Scope held

Articulation and voice-character controls are **written and deliberately
unapplied**, held out of the tree so they cannot touch this comparison. The
mechanical "na", articulation and character remain retained and unacted on,
awaiting Krish's listening. No further model run is authorized or requested, and
no weights were trained: this pass changed one gain.
