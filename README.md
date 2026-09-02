# Raaga AI Music Composer

A Windows **desktop** music-director workstation, written in Python with
PySide6/Qt. Not a web app, not a browser tool, and not a
prompt-in / random-song-out generator: the creator directs the work, and the
system carries it out instrument by instrument, section by section, time range
by time range.

Implements *Master Specification v0.2 - Desktop Python Edition* and the
*Raaga Self-Learning Music Agent* specification v1.0.

Behind the composer sits a student: a persistent agent with an embedded
curriculum, its own ears, permanent memory and a teacher. It listens, practises,
is marked, remembers what it learned after a restart, and the music the
application writes changes because of it.

---

## What it does

The workflow the application is built around:

```
creative direction -> raaga -> TUNE -> lyrics fitted to the tune -> voice
   -> studio vocal-only master -> interactive arrangement -> mix -> master
```

* **Creative brief** in ordinary words - "lonely, late at night, but still
  warm" - not a formal music prompt.
* **Raaga selection** with reasons, alternatives, comparison and locking. Raaga
  knowledge is structural data (arohanam, avarohanam, jeeva and nyasa swaras,
  prayogas, gamaka guidance, moods), not something a language model is trusted
  to remember.
* **Tune first.** The melody is composed on the raaga's own ladder, so ascending
  motion uses arohanam notes and descending motion uses avarohanam notes.
  Versions, variations, per-section regeneration, tempo changes and locking.
* **Lyrics fitted to the tune** - exact syllable counts, stress on long notes,
  breath positions from the melody's own phrasing. Edit or rewrite one line
  without touching the locked tune.
* **Voice** - built-in singer profiles, or your own profile derived from
  recordings you supply. Vocal direction (soft, strong, emotional, sad,
  devotional...) with dynamics, vibrato, breath and sustain.
* **Studio vocal-only master** - "give me the song without instruments"
  produces a fully produced vocal: cleanup, EQ, compression, de-essing,
  ambience, stereo treatment, loudness normalisation and limiting. Not a dry
  demo.
* **Interactive arrangement** - add, remove, replace and re-time instruments
  over exact ranges. Every part is derived from the approved tune: same tonal
  centre, same raaga, same tempo, register chosen to stay out of the vocal's
  way.
* **Continuous voice control** with barge-in: start speaking and playback
  pauses and cancellable work stops before you finish the sentence. The newest
  instruction always wins; stale results are discarded.
* **Non-destructive throughout** - versions for tune, lyrics, takes,
  arrangements and mixes; region and section locks that are enforced, not
  advisory; undo/redo; autosave; crash recovery from a backup copy.

## The student behind the instrument

Open the **Learning** tab. Press **One lesson** to watch it study, or **Start
learning** to let it work in the background while you compose.

* **A curriculum it actually executes.** Thirteen foundation lessons - tone
  from noise, higher from lower, finding Sa, naming a swara, ascending from
  descending, intervals, pulse, holding and reproducing a pattern, inventing
  one, varying one - then twenty-two lessons per raaga from identity and
  arohanam through characteristic phrases, grammar and drift, up to alapana,
  kalpana swara and composing to a mood. Then cross-raaga comparison.
* **Real ears.** Every listening lesson synthesises a phrase, analyses the
  audio it just made - pitch contour, tonic, swaras, phrase boundaries, tempo -
  and compares. A lesson passes only if the hearing and the knowledge both
  work.
* **Permanent memory.** One SQLite file holds every source, phrase, fact,
  score and correction, with provenance and confidence on all of it. Close the
  app and it carries on from where it stopped.
* **A teacher.** Twelve separate scores - swara correctness, raaga
  correctness, phrase authenticity, drift, rhythm, coherence, originality,
  mood, brief, structure, interest, expressiveness - plus what went wrong and
  one thing to do about it. Press **Mark the current tune** to see it judge its
  own work.
* **Grammar, not songs.** Learned phrases are fingerprinted; a tune that
  repeats too long a run of one is thrown away and rewritten.
* **Your corrections count.** Say *"This does not sound like Keeravani"* and it
  lowers its confidence in the phrases it leaned on, and stops using them.

### Teaching it from your own recordings

**Choose my learning folder...** points it at audio you are entitled to use.
Name the files or folders after the raaga - `Keeravani-alapana.wav` - so it
knows what it is hearing. It will analyse them, extract phrases, and record
where every fact came from.

With no folder chosen it still learns: it renders exercises from its own
structural library and listens to itself, the way a student plays a scale to
hear it. Nothing is fetched from the internet. The optional web provider is off
by default and, when enabled, only writes down *leads* for you to follow - it
does not download anything, and it never touches a paywall or an access
control.

## Install and run

```
setup.bat        (once - creates .venv and installs dependencies)
run.bat          (launches the desktop application)
```

Or by hand:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m raagacomposer
```

Requires Python 3.10+ on Windows. Everything below works with **no API keys and
no network**.

### Packaged build

```
packaging\build_windows.bat
powershell -File packaging\make_shortcut.ps1
```

Produces `dist\RaagaComposer\RaagaComposer.exe` - a windowed application launched
from a desktop or Start-menu shortcut, with no terminal.

## Optional providers

The app is complete without them; they raise quality where configured.

| Capability | Package | Credential |
|---|---|---|
| Continuous speech-to-text (offline) | `vosk` + a model in `%APPDATA%\RaagaComposer\models\` | none |
| Speech-to-text (offline, batch) | `faster-whisper` | none |
| Claude, for lyrics and raaga reasoning | `anthropic` | `ANTHROPIC_API_KEY` |
| A small local model, in process | `llama-cpp-python` + a `.gguf` in `%APPDATA%\RaagaComposer\models\llm\` | none |
| A small local model, via a server | Ollama running on `127.0.0.1:11434` | none |
| MP3 export | `ffmpeg` on PATH | none |

Keys are read from the environment first, then
`%APPDATA%\RaagaComposer\credentials.json`. Nothing is hard-coded, and every
provider sits behind a replaceable interface in `raagacomposer/providers/`.

Without a speech backend the microphone panel still works: type an instruction
and it goes through the identical interpretation pipeline.

## Which model answers what

The application asks a language model for five things, and they are not alike.
One is creative and tightly constrained; one happens while the creator is still
speaking. So there is no single "the model" - there is a router
(`providers/router.py`) that picks per task on three things, in this order:

* **complexity** - a hard task goes to the strongest backend that is up
* **cost** - an easy one goes to the cheapest, and local models are free
* **offline availability** - anything unreachable is out of the running, and a
  backend that fails mid-request hands the task to the next rather than
  losing it

With Claude configured and a local model running, that works out as:

| Task | | Goes to |
|---|---|---|
| Write lyrics to the tune | hard, quality is heard | Claude Opus 5 |
| Rank raagas for a brief | hard, quality is heard | Claude Opus 5 |
| Answer a musical question | middling | Claude Haiku 4.5 |
| Choose instruments for a feel | easy | the local model |
| Classify a spoken instruction | easy, and urgent | the local model |

A small local model is only offered the work it can actually do. Measured here
on `llama3.2:3b`, CPU only: it classifies an instruction, picks instruments and
answers a question in 6-46 seconds, but given ten lyric lines it ran for **704
seconds and returned nothing usable**, and when it did answer it wrote in Tamil
script rather than the transliteration the synthesiser can sing. The built-in
lexicon engine fits all ten exactly, instantly. So the two tasks whose quality
is heard - lyrics and raaga choice - skip any backend below a capability floor
rather than waiting on it. Raise `llm_local_strength` if your local model is
genuinely bigger, and it will be offered them.

The live chain for every task is printed in the output panel, under
`Help -> About`, and in `Help -> Export diagnostics...`, so what answered what
is never a guess. Change it with `llm_routing` in settings: `auto`,
`local_first`, `claude_first`, `local_only` (nothing leaves the machine),
`claude_only`, or `off`.

**Everything below is still true with none of it installed.** No key and no
local model means every task falls to the built-in rule and lexicon engines -
the default path, not a degraded one. The whole test suite runs that way.

### Turning Claude on later

Nothing needs rebuilding or editing. Either:

```
setx ANTHROPIC_API_KEY sk-ant-...
```

or add `{"anthropic_api_key": "sk-ant-..."}` to
`%APPDATA%\RaagaComposer\credentials.json`. The router re-reads whether a key
exists before each request and rebuilds its backends when that changes, so a
running application picks it up within `llm_refresh_seconds` (30 by default).

### Turning a local model on

Either route; neither needs a key and neither touches the network.

```
ollama pull llama3.2:3b
```

or drop a `.gguf` into `%APPDATA%\RaagaComposer\models\llm\` and

```
.venv\Scripts\python.exe -m pip install llama-cpp-python
```

If Ollama is running but the named model is not pulled, it says so and names
the command - it will not answer with a model you did not ask for.

## Talking to it

```
"Play the first minute."
"Play from the second minute to the third minute."     -> 01:00-03:00
"Play the end."   "Play the last 30 seconds."
"Play from the chorus."   "Start five seconds before this point."
"Add veena here."
"Use saxophone for this interlude."
"Bring strings after the chorus."
"Take the drums out here."
"Use only piano for the first 15 seconds."
"Replace violin with veena."
"Make this part lighter."
"I want this to feel lonely, late at night, but still warm."
"Give me the song without instruments."
"Lock the pallavi."   "Undo."   "Mix the song."

"Learn Keeravani."                    "Study more alapana examples."
"Why did you choose this phrase?"     "What are you learning?"
"This does not sound like Keeravani."
```

If you name an instrument the catalog does not have, it says so and offers the
closest available ones. It never quietly substitutes a different instrument.

## Layout

```
raagacomposer/
  app.py            application controller - the only place project state changes
  core/             data model, persistence, jobs, settings, undo, locking, logging
  raaga/             raaga knowledge store (data/raagas.json) and selection engine
  music/            theory, structure, melody, validator, instruments, synthesis,
                    arrangement, mixer
  lyrics/           syllable fitting engine and lyric generation
  voice/            singer profiles, singing synthesis, vocal mastering
  audio/            DSP, playback engine, export (WAV/MP3/MIDI/MusicXML/stems)
  speech/           microphone capture, speech adapters, intent and timeline parsing
  providers/        provider abstraction and adapters
  ui/               PySide6 desktop UI: main window, timeline widget, panels
tests/
  unit/             one module at a time
  integration/      subsystems together, through the real controller and window
  regression/       one test per defect found and fixed, plus golden output
  golden/           pinned deterministic output
packaging/          PyInstaller spec, Windows build and shortcut scripts
```

## Tests

685 tests in three suites, all run with pytest.

```
tests\run_all.bat                 everything (about five minutes)
tests\run_fast.bat                unit + regression only (about ten seconds)
```

or directly:

```
.venv\Scripts\python.exe -m pytest tests\unit -q          501 tests
.venv\Scripts\python.exe -m pytest tests\integration -q   143 tests
.venv\Scripts\python.exe -m pytest tests\regression -q     41 tests
.venv\Scripts\python.exe -m pytest tests -q -m "not slow"
```

**Unit** - no I/O beyond a temp directory, under three seconds for the lot:
pitch and time helpers, the data model and its JSON round trip, the raaga
library and selection engine, song structure, melody generation and the raaga
validator, syllable fitting and lyric writing, the instrument catalog and
synthesis, DSP and the vocal mastering chain, the timeline parser and intent
interpreter, conversation state, the job manager, undo and lock protection,
settings and credentials, project storage, the orchestration and mix engines,
voice profiles, singing synthesis, providers, and every export format.

**Integration** - subsystems working together through the real controller,
with real synthesis and real files on disk:

* `test_acceptance_scenario.py` - specification section 21 step by step, one
  test per numbered step, from "create a project" through to "reopen it and
  confirm the accepted tune, lyrics, voice, arrangement and history remain".
* `test_controller_flows.py` - stage progression, versions, locking, exports,
  autosave, diagnostics, barge-in.
* `test_voice_commands.py` - spoken instructions changing the project, and the
  conversation record of what was applied, ignored or refused.
* `test_restart_recovery.py` - restart, crash mid-save, missing audio file,
  failing provider, absent audio device.
* `test_ui_window.py` - the real Qt window and every panel, driven offscreen,
  writing screenshots for inspection.
* `test_agent_acceptance.py` - the learning specification's own tests A to F:
  persistence across a restart, ranked raaga suggestions with reasons and
  confidence, stating what it learned about Keeravani, a prelude that is real
  music rather than one sustained note, a full learning cycle with provenance,
  and a correction that lowers confidence and changes what it plays next.
* `test_agent_in_the_app.py` - the agent behind the existing interface: Apply
  Brief answered from memory, Generate Tune composing from learned phrases,
  learning controlled from the UI, and the whole original workflow still
  running.

**Regression** - each test names a defect that was actually found here and
fixed, so a failure explains itself: a control character that silently killed a
regex, a de-esser that boosted sibilance instead of ducking it, a number word
matching inside another word, a substring match that returned a female voice
when asked for a male one, an uninterruptible job reported as cancelled rather
than stale, a bare "add veena" pinned to one section, a playhead that could not
move before the first render, and a window that demanded 2700 pixels of width.

`test_golden_output.py` pins the deterministic output of the melody, structure,
timeline, intent and lyric engines against files in `tests/golden/`. Approve an
intended change with

```
set RAAGA_UPDATE_GOLDEN=1
.venv\Scripts\python.exe -m pytest tests\regression
```

and review the diff before committing it.

## Where things stand

Development order is `COMPLETE -> WORKING -> RELIABLE -> CORRECT -> EXCELLENT`.
Every subsystem in specification section 12 exists and is connected; the whole
workflow runs end to end; reliability, locking, cancellation, stale-result
rejection and restart recovery are covered by tests. Audio quality, instrument
realism, lyric craft and orchestration intelligence are the EXCELLENT phase and
are deliberately not finished yet - see `docs/DECISIONS.md`.
