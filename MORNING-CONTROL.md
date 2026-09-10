# Morning control — preferred 04 reproduced exactly

One conversion, control first. Contract `46e5ca2` (v2). Output at
`renders/morning-control/`, written once, nothing in the historical directory
touched.

## Receipt

| | |
| --- | --- |
| **PID** | **24504** |
| Command | `…/Raaga-Singing-Experiment-20260908/.venv/Scripts/python.exe -B -m trials.morning_control` |
| Launch → finish | `2026-09-10T01:14:04Z` → `01:17:09Z`, 184.6 s |
| Exit code | **0** |
| Runner | `trials/morning_control.py` |

The PID, command, both timestamps and the exit code are in the run's own
`control-status.json`, written before the model loaded and updated at each stage
— not reconstructed afterwards, and not dependent on my terminal capture. The
previous two runs lost their PID to exactly that mistake.

## Inputs, verified before and after

| file | sha256 |
| --- | --- |
| `05-recorded-voice-word-timing-lower-register.wav` | `e650a658…` |
| `spb-credited-reference-8s.wav` | `7ba960f2…` |
| DiT checkpoint | `42aef93f…` |
| `local-config.yml` | `fbca8cca…` |
| preserved 04 raw | `c1669e72…` |
| preserved 04 listening | `1e9da45d…` |

All six matched before launch and again after. Nothing was written into
`Raaga-Vocal-Quality-20260909`.

## Settings, faithful to the morning run

CFG **0.3**, 30 steps, length 1.0, F0 conditioning on, auto F0 adjust off,
semitone shift 0, fp16 off, CPU float32, seeds 20260909 for torch, random and
numpy, `torch` 4 threads with 1 interop, raw saved as **FLOAT**. Offline, no
downloads, no training.

The runner is a copy of `probe_spb_conversion_core.py --reduced-guidance` changed
only in where it writes, with `QUALITY` and `ROOT` bound explicitly so that a
copy could not silently repoint the historical inputs at this workspace.

## Result — the converter reproduces exactly

| | |
| --- | --- |
| Max absolute sample difference | **0.000000e+00** |
| RMS difference | **0.000000e+00** |
| Correlation | **1.000000000000** |
| Frames / rate | 199168 / 44100, identical |
| Peak | 0.544607 both |
| RMS | −17.1903 dBFS both |
| Samples at or beyond full scale | **0** |

**The listening copy is byte-identical to the preserved 04**: `1e9da45d…`,
produced with the preserved scalar gain of **−0.779490993561403 dB** and PCM16 —
the recorded scalar reused, explicitly *not* a fresh loudness match. The
historical listening RMS was −17.96973386044995 dBFS, not exactly −18.

The raw FLOAT files differ in **three bytes at offsets 60–62**, which are the
WAV `PEAK` chunk's write timestamp (`0x6aa1882c` against `0x6aa20514`). The peak
value in that chunk is identical, and the files are byte-identical from offset
100 onward. **Audio data is bit-for-bit the same;** only the moment of writing
differs.

## What this establishes, and what it does not

**Establishes:** the morning converter, its settings and its runtime are intact
and reproducible on demand. Nothing was replaced, lost or silently changed. That
question is closed.

**Does not establish:** anything about why Krish prefers the morning result.
His 70% against 30% is a listening recollection, not a measured score, and this
run produces no acceptance of any kind. The comparison here is numerical
reproduction, not quality.

Worth stating plainly, since it is now measured on both sides: morning 04 peaks
at 0.545 with **no** samples at full scale, while current C peaks at 1.0 with 117
at the rails. The two families also differ in words, performance source, melody,
register and duration — so that level difference is one observation among several
and is not offered as the reason for anything.

## Stop

Stopping here for independent reproducibility review and listening, as
instructed. No second render, no transposition, no isolated tail, no current
lyrics. The current-lyrics trial remains a separate step, and the distinction
holds: the converter is recoverable, but a human performance of different words
cannot be produced by swapping a filename.
