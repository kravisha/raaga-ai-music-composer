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
Name the files or folders after the raaga - `Keeravani-alapana.wav`, or a
folder called `Keeravani` - so it knows what it is hearing; a file whose name
and folder mention no raaga is not picked up by any of them. Aliases count, so
`Yaman` is read as Kalyani. Audio only: `.wav`, `.flac`, `.ogg`, `.aiff`,
`.mp3` - a video file is ignored, so extract its audio first. Only the first
two minutes of each file are analysed, so several short clips teach it more
than one long class.

A real recording is not a rendered exercise, and it is not treated as one. A
lesson is somebody talking, over a tanpura or shruti box that never stops,
occasionally singing - so before the ears hear it:

* **The drone is found and taken out**, and its fundamental becomes Sa. A
  tanpura exists to declare the tonic, so it is believed rather than guessed
  at. A long held note is *not* mistaken for a drone: a drone sounds for the
  whole recording and brings a chord of partials, where a held note brings only
  itself.
* **The talking is silenced.** Singing holds a pitch - where it is ornamented
  it oscillates around one - and speech glides and never settles. That is the
  whole test: no language model, and nothing that needs to know Telugu or
  Tamil. Gamaka swinging well over a semitone still reads as singing.

Silenced stretches are muted in place, never cut out, so every timestamp still
means what it meant. If the gate cannot make sense of a recording it says so
and leaves the audio alone rather than handing back silence. It will analyse
what remains, extract phrases, and record where every fact came from.

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
| Richer lyric writing, fuzzy intent | `anthropic` | `ANTHROPIC_API_KEY` |
| MP3 export | `ffmpeg` on PATH | none |

Keys are read from the environment first, then
`%APPDATA%\RaagaComposer\credentials.json`. Nothing is hard-coded, and every
provider sits behind a replaceable interface in `raagacomposer/providers/`.

Without a speech backend the microphone panel still works: type an instruction
and it goes through the identical interpretation pipeline.

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

650 tests in three suites, all run with pytest.

```
tests\run_all.bat                 everything (about five minutes)
tests\run_fast.bat                unit + regression only (about ten seconds)
```

or directly:

```
.venv\Scripts\python.exe -m pytest tests\unit -q          458 tests
.venv\Scripts\python.exe -m pytest tests\integration -q   149 tests
.venv\Scripts\python.exe -m pytest tests\regression -q     43 tests
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
