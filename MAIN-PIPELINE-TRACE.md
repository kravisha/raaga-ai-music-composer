# Main Raaga pipeline — swara → lyrics → voice, traced

Read-only, on `raaga-ai-music-composer` at `c36dc22` (main). No render, download,
training or new task. **This is the production runtime, not our isolated
`nilavaai` fixture** — §5 separates them.

## 1. The chain, by file and function

| stage | file | function |
| --- | --- | --- |
| Notes with swara, duration, gamaka | `raagacomposer/core/models.py:102` | `class Note` |
| Syllable slots per note group | `raagacomposer/lyrics/fitting.py:49` | `build_slots(melody, ...)` |
| Word → syllables | `raagacomposer/lyrics/fitting.py:72` | `syllabify(word)` |
| Provider boundary | `raagacomposer/providers/local.py:41-53` | `LocalVoiceProvider.render_vocal` |
| Melody+lyrics → segments | `raagacomposer/voice/renderer.py:346` | `render_melody` |
| **The binding** | `raagacomposer/voice/renderer.py:103` | `plan_segments` |
| **The phoneme step** | `raagacomposer/voice/renderer.py:88` | `split_syllable` |
| Audio | `raagacomposer/voice/renderer.py:159` | `render` |

`LocalVoiceProvider.status()` returns, in its own words, **"formant singing
synthesiser."**

## 2. Alignment already exists — I must not have implied otherwise

`Note` carries `swara`, `midi`, `start`, `duration`, `velocity`, `gamaka`,
`syllable`, `section_id`. `build_slots` groups notes and records
`note_indices`, `durations` and `stresses`, so a lyric line states **exactly
which notes its syllables land on**. `plan_segments` then reads that mapping
directly:

```python
for line in lyrics.lines:
    for idx, syl in zip(line.note_indices, line.syllables):
        syllable_for[idx] = syl
```

**Raaga already selects the raga, generates swaras with unequal durations, fits
lyrics to specific notes, and hands all of it to the voice.** Note-level
syllable alignment is present in production and is not the gap.

## 3. What actually reaches synthesis

Per note, `plan_segments` builds a `SungSegment` carrying `start`, `end`,
`midi`, `syllable`, `vowel`, `consonant`, `velocity`, `gamaka`, `legato`.

**Duration and pitch controls arrive intact.** Gamaka is honoured with real
gestures — `renderer.py:237-248`: `kampita` becomes a 5.5 Hz oscillation,
`slide_up` and `slide_down` become leading portamento over the first 30% of the
note. Plus vibrato, jitter and legato.

**The phoneme controls are where it narrows, sharply.** `split_syllable` reduces
each syllable to **one onset consonant and one vowel**:

- **11 vowels** (`a aa e ae i ee o oo u ai au`) plus a hum. No length contrast
  beyond what those spellings imply, and no `ii`/`uu`.
- **Consonants by class only** — `PLOSIVES = kgtdpb ch j tt dd`,
  `FRICATIVES = sfhvz sh th`, `NASALS = mn ng ny`, `LIQUIDS = lrywv`. Each class
  gets one generic burst or noise shape (`_add_consonant`), so `k` and `p` are
  the same 28 ms burst at 4200 Hz.
- **Codas are dropped entirely.** `split_syllable` returns onset + vowel; the
  final consonant of `வந்` or `தால்` never reaches audio.
- Input is `re.sub(r"[^a-zA-Z]", ...)` — **Tamil script is stripped**, so
  production expects Roman transliteration.

## 4. The exact missing articulation step

Everything upstream is intact: raga, swaras, unequal durations, note-level lyric
fitting, gamaka, dynamics. The break is at one point:

> **`split_syllable` collapses a Tamil syllable to (one onset class + one of 11
> vowels), and `_add_consonant` renders that class as a generic burst. The coda
> is discarded and consonant identity within a class is lost.**

That is the missing articulation step, and it is a **representation** gap before
it is a model gap: even a better synthesiser would receive `("d", "aa")` for
`தால்` — no retroflex/dental distinction, no `l` coda.

To close it, the interface between `plan_segments` and `render` must carry a
richer phoneme unit — onset, nucleus **with length**, and coda — and the
synthesis side must have something to do with it.

## 5. Production versus our fixture — the boundary

| | main Raaga | our `nilavaai` fixture |
| --- | --- | --- |
| Notes | `Note` objects from the raga engine, unequal durations, gamaka | hand-authored `Event`s with fractional MIDI and bends |
| Lyric binding | `build_slots` → `note_indices` | `align.place` over sung stretches |
| Phonemes | `split_syllable`: onset class + 1 of 11 vowels, coda dropped | `tamil_frontend`: Tamil script, vowel length, digraphs, codas, refusal by name |
| Synthesis | formant resonators, in-process | Hanakoloid CVVC samples + Worldline DLL |
| Conversion | none | Seed-VC, offline |

**They are different systems.** Our fixture never touched production code, and
nothing in this trace changes that. The fixture's Tamil frontend is strictly
richer at the phoneme step; the production engine is strictly richer upstream —
it *generates* the raga, swaras and fitting that we hand-authored.

## 6. Narrowing my earlier conclusions

**"No synthesis capability is installed" was wrong.** `VitsModel` and
`VitsTokenizer` import cleanly from `transformers 4.57.6` in the
Singing-Experiment venv on `torch 2.7.1+cu126` — I verified it. I searched for
standalone TTS packages and missed that `transformers` ships VITS. **What is
missing is Tamil weights, not synthesis code.** Corrected.

**The Tamil human unit bank and the lowered Japanese bank are hypotheses, not
proven fixes.** Neither has been tested and I should not have ranked them as
solutions.

**The 132 Hz figure is not SPB's vocal range.** It is a median over an 8-second
*song mix with accompaniment*, so instruments are in it. It does not establish
his range, and the +22-semitone gap I computed from it is correspondingly
uncertain.

Morning 04 stays the baseline; E stays rejected.

## 7. The concrete next engineering step

**Widen the phoneme representation at the `plan_segments` → `render` boundary
before choosing any model.** Concretely: carry `(onset, nucleus, coda)` with
length preserved instead of `(consonant, vowel)`, and accept Tamil script rather
than stripping it.

That step is model-independent — it is required whether the eventual voice is
the formant renderer, a unit bank, or VITS with Tamil weights — and it is the
one place where the production pipeline currently loses information it already
has. Everything else upstream is already doing its job.

**Nothing implemented. No model chosen. No render.**
