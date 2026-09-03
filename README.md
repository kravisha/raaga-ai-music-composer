# Raaga AI Music Composer

A Windows **desktop** music-director workstation, written in Python with
PySide6/Qt. Not a web app, not a browser tool, and not a
prompt-in / random-song-out generator: the creator directs the work, and the
system carries it out instrument by instrument, section by section, time range
by time range.

Implements the *Canonical Implementation Specification v0.3*
(`docs/spec/CANONICAL_SPEC_v0.3.txt`, 2026-09-03), which consolidates the
earlier *Master Specification v0.2 - Desktop Python Edition*, the *Raaga
Self-Learning Music Agent* specification, the Training-tab specification and
the Knowledge Base architecture. Code comments still cite the earlier
documents' section numbers; `docs/spec/CROSSREF.md` maps each to v0.3, and
`docs/PLAN_v0.3.md` records what v0.3 changed and its status.

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

Click **LEARN** in the toolbar (or `Ctrl+2`) to switch out of the composer
into the agent's own workspace - a full screen, not a tab squeezed in beside
Tune and Lyrics. Its Dashboard area has **Start** and **One lesson** buttons
to watch it study, or let it work in the background while you compose in
**MAIN** (`Ctrl+1`).

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

### Training Sources, in LEARN

A second way to teach it, and the one that scales: **search for material,
choose what it may learn from, and read what it made of each source.** This
lives in LEARN's **Training Sources** area (formerly its own "Training" tab).

1. Type what you are after - *Kamboji raga beginner lesson*, *Carnatic gamaka
   techniques*, *Keeravani arohanam avarohanam tutorial*. Raaga names are
   matched across spellings, so *Kamboji*, *Kambhoji*, *Yaman* and *Bhoop* all
   find the right one.
2. Press **Search**. About ten candidates come back, each showing what it is,
   how relevant it looks, and - the column that matters - whether its content
   can actually be reached.
3. **Tick the ones it may learn from.** Nothing is ticked for you. Searching is
   not approving, and the system never learns from a result you did not choose.
4. **Add to learning queue**, then **Start learning**. One source at a time,
   with the stage it has reached shown as it goes.
5. Read the **Learning Report**: what the source contained, what it understood,
   what it *learned* (kept separate), what it already knew that this confirmed,
   anything that contradicted what it held, what this changes about the music,
   and what to study next.

Every learned item keeps its provenance - which source, which run, which
objective, which timestamp, what evidence, what confidence - and LEARN's
**Knowledge** area will show you all of it for any item you select. You can
mark an item incorrect or approve a disputed one; nothing important ever
changes without leaving a trace.

**What it will and will not do.** It does not download from the internet. A
source on the network is listed honestly as `Metadata only` and its report says
`METADATA ONLY - CONTENT NOT ANALYZED` rather than pretending it sat through
the lesson. Nothing here logs in, follows a paywall, or works around a
download protection. When a source is out of reach it offers you the two ways
forward that are actually yours to take: **supply the file** or **provide the
transcript**. Do either and the same source becomes a real lesson.

There is also a **third** kind of source, and it is the one that works on a
machine with no network and nothing of your own: exercises the system builds
from its own raaga library, plays, and then listens back to. A student playing
a scale to hear it.

**Heard is not the same as read.** A phrase it tracked and identified by ear
becomes something the composer can quote. A phrase a teacher merely *stated* in
a transcript is recorded for you to see but deliberately withheld from the
music, because nothing has verified it. The report says which is which.

### What it knows, and where it got it

Behind every area of LEARN is one **Knowledge Base** - a single
SQLite file that is created once and then grows. It is not recreated when you
restart, upgrade or reinstall, and a damaged one is never quietly replaced with
an empty one: it is kept, a copy is preserved, and you are told.

Knowledge in it is a network rather than a list. Ask about Kambhoji and you
reach not only its definition but its characteristic phrases, its gamakas, the
things it must *not* do, the examples that demonstrate it, and the sources
behind each of those - because they all hang off Kambhoji as connected nodes.
Spelling does not matter: *Kamboji*, *Kambhoji* and *Khamaj* all arrive at the
same entity.

Four things it will not do:

* **It will not lose a source.** Every learned claim carries which source, which
  learning run, where in that source, what evidence, and whether it was heard,
  read or inferred. Ask any item where it came from and it can tell you.
* **It will not pile up copies.** The same fact from ten teachers is one
  canonical claim with ten evidence records, and confidence that rises because
  the sources are independent - not ten rows.
* **It will not overwrite a disagreement.** When a source contradicts something
  held, both are kept, the conflict is recorded with a recommendation, and a
  person decides. Music disagrees for good reasons and "context-dependent" is
  one of the answers you can give.
* **It will not turn a guess into a fact.** Confidence is stored with the
  reasoning that produced it, so a score can be read back as a sentence -
  *"0.63 from direct demonstration +0.20, independent sources +0.15..."*

You can mark anything incorrect, approve a disputed claim, or adjust its
confidence, and the previous reading is kept with a note of who changed it and
why. Nothing important ever changes without leaving a trace.

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

### Settings -> Providers

The key can also be entered from inside the application: Settings has a
Claude group with fields to enter, validate, change and remove it, and a
Providers table showing every backend's live status (Configured / Not
configured / Unavailable for Claude, Available / Not installed / Ready for
local models) alongside the routing policy. With the `keyring` extra
installed (`pip install raaga-composer[secure]`, or just `pip install
keyring`), a key entered there goes into the OS credential store - Windows
Credential Manager on this machine - rather than `credentials.json`; without
it, the file is used automatically. The environment variable still wins over
both and is read-only from the dialog, so `setx` keeps working exactly as
above.

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
  agent/            the student: memory, curriculum, ears, practice, critic
  training/         the Training tab: search providers, access policy, ingestion
                    pipeline, objectives, validation, queue
  kb/               the Knowledge Base: the durable knowledge network, its
                    normalization, confidence model, hybrid retrieval, context
                    builder, librarian and migrations
  providers/        provider abstraction and adapters
  ui/               PySide6 desktop UI: main window with two top-level
                    workspaces (MAIN, the composer; LEARN, the agent's own
                    screen - see learn_workspace.py), timeline widget, panels
tests/
  unit/             one module at a time
  integration/      subsystems together, through the real controller and window
  regression/       one test per defect found and fixed, plus golden output
  golden/           pinned deterministic output
packaging/          PyInstaller spec, Windows build and shortcut scripts
```

## Tests

913 tests in three suites, all run with pytest.

```
tests\run_all.bat                 everything (about five minutes)
tests\run_fast.bat                unit + regression only (about ten seconds)
```

or directly:

```
.venv\Scripts\python.exe -m pytest tests\unit -q          657 tests
.venv\Scripts\python.exe -m pytest tests\integration -q   197 tests
.venv\Scripts\python.exe -m pytest tests\regression -q     59 tests
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
  writing screenshots for inspection; includes TEST I, that LEARN is a
  top-level workspace and not a composer tab.
* `test_learn_workspace.py` - LEARN's six areas offscreen: the Dashboard
  against the real agent, Start/Stop reaching the real background learner,
  one real practice exercise on the job manager, and the Knowledge area's
  search and provenance widgets.
* `test_agent_acceptance.py` - the learning specification's own tests A to F:
  persistence across a restart, ranked raaga suggestions with reasons and
  confidence, stating what it learned about Keeravani, a prelude that is real
  music rather than one sustained note, a full learning cycle with provenance,
  and a correction that lowers confidence and changes what it plays next.
* `test_agent_in_the_app.py` - the agent behind the existing interface: Apply
  Brief answered from memory, Generate Tune composing from learned phrases,
  learning controlled from the UI, and the whole original workflow still
  running.
* `test_training_acceptance.py` - the Training specification's section 19
  demonstration, one test per numbered step, from typing the search phrase
  through to closing the application and finding the history and the learned
  knowledge still there.
* `test_kb_acceptance.py` - the knowledge-base specification's own ten
  acceptance tests, numbered as it numbers them: knowledge surviving a
  restart, a fact naming its source and run, one canonical fact with two
  evidence records, a contradiction recorded rather than overwritten,
  retrieval that answers what composing actually needs, a correction and its
  history surviving a restart, committed knowledge surviving an abrupt
  termination, a training run integrating into the Knowledge Base, Compose
  reading through the service with a retrieval trace, and a restart plus
  schema migration leaving everything intact.
* `test_training_flow.py` - what the specification spends most of its words
  on: a source that cannot be fetched saying so, supplying the file or the
  transcript turning it into a real lesson, relearning without destroying the
  earlier report, a failed source that still produces a report, and a run
  interrupted by a crash being returned to the queue.
* `test_preprocess_in_research.py` - a lesson recording dropped in the
  learning folder, prepared and then heard.

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
