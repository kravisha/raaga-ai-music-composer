# Capability inspection — what can actually change the sung words

Read-only. No render, no model run, no download, no training, no new task.
Bounded to: what is installed that can turn Tamil text and a melody into sung
audio, and what existing material could do it differently from the rejected
Japanese-CVVC path.

## 1. Installed text-to-speech / singing capability: none

Checked all three virtualenvs and the system path for `TTS`, `coqui`, `piper`,
`espeak`, `phonemizer`, `epitran`, `g2p`, `indic-*`, `ai4bharat`, `nnsvs`,
`diffsinger`, `utaupy`, `pyopenjtalk`, `bark`, `tortoise`, `so-vits`, `vits`,
`espnet`, `gruut`, `openvoice`, `melo`, `edge-tts`, `xtts`, and for the
`espeak` / `espeak-ng` / `festival` / `flite` / `mbrola` binaries.

**Nothing is installed.** There is no text-to-speech and no singing-voice
synthesiser on this machine.

Also absent: **separation** (`demucs`, `spleeter`, `openunmix`) and any **forced
aligner** (`montreal-forced-aligner`, `aeneas`). Present but unused:
`faster-whisper 1.2.1` in `cinema/.venv` — ASR, useful for labelling, not for
synthesis.

## 2. What can produce sung audio from text and a melody

Three paths exist. **All three originate a synthetic carrier.**

| path | what it is | evidence |
| --- | --- | --- |
| Hanakoloid + Worldline | Japanese CVVC sample bank, 758 aliases, driven by the Worldline DLL | this is E's path — **rejected** |
| `raagacomposer/voice/renderer.py` | **formant synthesis** — two-pole resonators at F1/F2/F3, `_resonator`, `_add_consonant`; one incidental match for `wav|sample|bank|corpus` in the whole file, so **no unit or sample use at all** | inspected |
| the `raga-voice.zip` carrier | formant synthesis, same class | reviewed earlier |

**Seed-VC cannot originate words.** It converts timbre and preserves f0 and
timing; it needs a guide that already says the words. That is the whole shape of
the problem.

## 3. The exact missing transformation

**Nothing installed turns Tamil text plus a melody into a *human-sounding* sung
performance.** Every available originator is synthetic, and Seed-VC inherits
whatever articulation the carrier gives it. Krish has now rejected that result
twice — A ("bad") and E ("bad, not acceptable") — from two different carriers and
two different guidance settings.

The missing piece is not a setting. It is a source of **real human Tamil
articulation** for words nobody has sung yet.

## 4. Existing material that could supply it

| pool | size | state |
| --- | --- | --- |
| SPB singing candidates | **36 files, 170.9 min** | **0 admitted**; 32 of 36 `co_credited_singing_candidate` (other singers present); song mixes, not isolated vocals |
| SPB interviews | 5 files, 96.9 min | 0 admitted; speech, not singing |
| Recorded-phrase material | ~2.5 min across all pools | small, and its **performer is not established** — see §6 |
| Hanakoloid bank | 758 aliases | Japanese, one speaker |

The inventory's own note is the right one: *"Raw runtime is not clean singing
time; Tamil phoneme/register/dynamic coverage and per-interval SPB identity
require review."*

## 5. Two materially different methods, and what each needs first

### A. Author the new tune in the reference's register — small, cheap, permitted

I measured this earlier: our guide sat at a **median 479.8 Hz** against the SPB
reference's **median 132.4 Hz** — about **+22 semitones**, with
`auto_f0_adjust=false` so nothing corrects it. We were asking for SPB's timbre
nearly two octaves above the register that reference demonstrates.

Krish's new direction says **change the lyrics and the tune**. Authoring the new
melody low — near C3–G3 rather than B♭4–D5 — is *not* the transposition that was
excluded; it is writing the new tune in a singable register from the start.

**What must change first, and the real risk:** the Hanakoloid samples are
recorded around **midi 71–72** (measured: `a` ≈ 534 Hz, `aa` ≈ 514 Hz). Asking
Worldline to render them two octaves down is a large stretch and may degrade
badly. **That is a genuine risk, not a formality**, and it is the thing to test
before believing this method.

### B. Replace the Japanese bank with Tamil human units — larger, and the real fix

Same architecture, different units: build a CV/VC unit bank from **real Tamil
singing** instead of a Japanese one. This attacks articulation directly, which is
Krish's first complaint.

**What must change before a single candidate:**

1. **Source admission** — 0 of 41. Needs identity verification and a source-use
   decision. **Krish's, not mine.**
2. **Separation** — not installed, and 32 of 36 files are co-credited mixes.
3. **Segmentation and labelling** into units with an `oto`-style map — needs
   phoneme-level alignment. `faster-whisper` gives words, not phonemes.
4. **Coverage** — the unit set must cover the vowels, codas and registers the
   new lyrics need, or it refuses like the current frontend does.

That is a real chain, and steps 1–3 are each substantial. I am not proposing it
as a next step this week; I am naming it as the method that would actually
address the complaint.

## 6. A correction I owe

I repeatedly described the morning guide as **"Krish's own recorded voice."** The
evidence does not support that. The attribution came from a filename
(`krish-repeated-phrase-native.wav`) and a hardcoded string in an older script,
and the chain traces to `WhatsApp Audio 2026-09-08 at 7.00.14 PM.mp4`. **A
filename is not evidence of who performed** — the standard I applied to everyone
else's work today and failed to apply to my own. The performer is not
established, and root is checking it.

Keeping three things apart from here on: the **input guide** (performer
unestablished), the **conditioning reference** (SPB-credited song excerpt), and
the **output** (SPB-conditioned synthesis — not an authentic recording).

## 7. Recommendation

**Method A is the only one launchable soon**, and only as a test of whether the
bank survives the stretch — not as a promise that it will sound better. Method B
is where the articulation problem actually gets solved, and it is gated on a
decision only Krish can make.

If both are unattractive, the honest third answer is that **this machine has no
way to make new Tamil words sound humanly sung**, and the gap is a capability
gap rather than a tuning gap. I would rather say that plainly than spend another
render finding it out.

**Nothing rendered. No candidate launched. Awaiting direction.**
