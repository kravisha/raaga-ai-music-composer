# Provisional decisions

Specification section 2 asks for reversible engineering decisions to be taken
without interrupting the creator, and documented. These are they. Each names
what would change if it is revisited.

## Platform

**PySide6/Qt for the desktop UI.** The spec's stated preference. A single
`QMainWindow` with docks and splitters; no browser, no web view, no Electron.
Reversing this means rewriting `raagacomposer/ui/` only - the controller
(`app.py`) has no Qt imports and the engines have no UI knowledge.

**Threads, not asyncio.** Generation and rendering run on a
`ThreadPoolExecutor`; completions are delivered to the UI thread by a queue the
Qt timer drains (`JobManager.drain`). Worker threads never touch Qt and never
mutate project state. This is simpler to reason about than mixing an asyncio
loop into Qt, and cancellation is explicit.

**sounddevice for playback and capture.** Gives a callback stream with real
position reporting, which the natural-language playback commands need. A
missing or failed device is reported, never raised into the UI.

## Music generation

**Local synthesis rather than a cloud music model, for version 1.** Every
instrument in the catalog can be heard immediately, offline, deterministically,
with no credentials. The spec's priority is COMPLETE before EXCELLENT, and a
cloud model cannot supply per-region regeneration that respects a locked tune
without significant extra work. `providers/base.py` defines
`MusicProvider`; swapping in a cloud renderer is one adapter class.

**Additive synthesis over an instantaneous-frequency curve.** Vibrato, gamaka
(kampita oscillation, slides) and legato glide become modulations of one pitch
curve rather than post-processing. This is what makes the Indian instruments
sound like themselves rather than like sampled organs.

**Source-filter singing synthesis.** A glottal source follows the melody's
pitch curve; formant resonators shape it into the vowel of each sung syllable;
consonants are short shaped transients. This is the honest weak point of the
current build - it sings the right notes with the right vowels and phrasing,
but it does not sound like a human singer. Replacing `providers/base.py`
`VoiceProvider` with an authorised singing-synthesis or voice-conversion
service is the intended EXCELLENT-phase upgrade, and needs no other change.

**Raaga knowledge as data, not prompt.** `raagacomposer/raaga/data/raagas.json`
holds 18 raagas with arohanam, avarohanam, jeeva and nyasa swaras, prayogas,
gamaka guidance, avoided phrases, moods and tempo ranges. A user file at
`%APPDATA%\RaagaComposer\raagas_user.json` merges in with the same shape, so the
library grows without touching code.

**Structure templates.** Film, devotional and simple templates scale their
cycle counts to the requested duration and drop optional sections when time is
short. A template is a list of slots in `music/structure.py`.

## Lyrics

**Two engines behind one call.** With a language model configured, lines are
requested with exact syllable counts and stress patterns, then re-fitted and
checked. Without one, a lexicon engine assembles lines from transliterated word
pools to the exact syllable count. The lexicon output is thematic rather than
poetic; that is a known limit, not an accident. The fitting engine - which is
the part the spec cares about - is identical either way.

**Transliterated Roman script.** Lyrics are written in Roman transliteration so
the singing synthesiser can pronounce them and so syllable counting is
mechanical. Native scripts would need a per-language grapheme-to-phoneme step.

## Interaction

**Rules first for intent, model second.** The rule tables in
`speech/intent.py` run on every utterance: fast, deterministic, no credentials,
and they never invent an instrument the creator did not name. A language model
is consulted only for sentences the rules cannot classify, and its answer is
still mapped onto the same closed intent set.

**"The Nth minute" convention.** "From the second minute" starts at 01:00 and
"to the third minute" runs to 03:00, per specification section 20 phase D. The
range is then clamped to the song's actual length.

**Barge-in cancels, it does not queue.** When speech is detected, playback
pauses and every cancellable job is cancelled immediately. Each job carries an
epoch per target; a result from a superseded epoch is discarded rather than
allowed to overwrite newer creator intent.

## Persistence

**JSON as the source of truth, SQLite as a journal.** `project.json` is written
atomically with the previous copy kept as `project.json.bak`; `project.db`
carries an append-only record of history, conversation, jobs, errors and
artifacts. A process killed mid-save leaves a readable account either way.
Audio artifacts live as files under the project folder and are never deleted by
an undo.

**Snapshot undo.** Undo stores whole-project snapshots rather than an inverse
command log. Heavier, but it cannot desynchronise - and losing an accepted tune
to a buggy undo entry is the failure the spec is most explicit about avoiding.

## Testing

**Three suites, separated by what they cost and what they prove.** Unit tests
touch one module and finish in seconds, so they are worth running on every
edit. Integration tests drive the real controller and the real Qt window with
real synthesis and real files, because the defects that mattered here appeared
only when subsystems met. Regression tests each name a defect that was
actually found and fixed, so a future failure explains itself rather than
needing archaeology.

**No mocks for the engines.** The generators, DSP and file handling are
deterministic and fast enough to exercise directly. Mocking them would test
the mock. The one substitution is the audio *device*, which is absent on a
build machine; the playback engine reports device failure rather than raising,
and that path is tested.

**Golden files for the deterministic output.** Melody, structure, timeline
parsing, intent classification and lyric assembly are all seeded and
reproducible, so their output is pinned in `tests/golden/`. A musical change
is legitimate; a silent one is not. `RAAGA_UPDATE_GOLDEN=1` re-writes them for
review.

**The spelling is "raaga" throughout** - package, modules, data files, API and
prose. It matches how the word is pronounced and how the project is named.

## The learning agent

Added 2026-08-31 from the self-learning agent specification. The composer is
unchanged as an instrument; the agent is the student behind it.

**Memory is one SQLite file** at `%APPDATA%\RaagaComposer\knowledge.db`,
written with WAL. It holds sources, learned phrases, raaga facts, curriculum
progress, compositions, feedback, queued tasks and an event log. It is the
source of truth; the in-memory views are rebuilt from it at every use rather
than cached, which is fast enough at this size and cannot drift.

**Nothing is overwritten.** Facts carry confidence and provenance. Two sources
that disagree are both kept and flagged, and the higher-confidence one is used
until a person says otherwise. An `observed_*` entry is evidence from one
recording rather than a claim about the raaga, so those never count as
disputes.

**The curriculum is executable data.** Every unit names a practice handler and
its parameters. Stage A is thirteen universal listening skills; Stage B is one
twenty-two-unit template instantiated per raaga, so a second raaga costs no new
curriculum; Stage C needs two raagas already taken to Stage B depth. A lesson
that beats the agent three times is rested and revisited later rather than
abandoned, with a hard cap on total attempts so it cannot loop.

**The agent learns by listening to itself first.** The research agent's
providers are, in order of preference: audio the creator puts in their own
learning folder (`user-supplied`); the application's own renders
(`own-output`); and reference exercises the agent renders from the shipped
structural library and then listens to with its own analysis pipeline
(`internally-generated`). That last one is the reason the whole loop works on
day one with no network and no rights questions at all: it is a student playing
a scale to hear it, not a copy of anyone's performance.

**The web provider records leads and fetches nothing.** It is off by default.
When enabled it writes down where authoritative material is said to exist,
marked `external-unverified`, and asks the creator to supply anything they are
entitled to use. Public availability is not permission, and no paywall, DRM or
access control is touched.

**Hearing is real, not simulated.** Pitch tracking is autocorrelation with
parabolic interpolation; the tonic is the pitch class that best explains the
rest of the material against the raaga's own intervals; notes are grouped from
the contour and phrases split at the breaths. Exercises synthesise a phrase,
analyse the audio, and compare - so a lesson can only pass if the ears and the
knowledge both work. Where a musician would be given the Sa by a tanpura, the
exercise supplies it: naming one note by ear is impossible without one.

**The critic never collapses its scores.** Twelve dimensions are reported
separately, with the mistakes it found and one piece of advice. A caller that
wants a single number asks for one, and the weighting it gets is visible in
`evaluator.py` rather than hidden.

**Originality is enforced, not hoped for.** Learned phrases are indexed by
octave-insensitive n-grams; a generated line sharing too long a run with one is
rejected and rewritten. Practice quotes at most three notes of an idiom, and
transposes it into the octave the line is already in.

**What the agent knows is what it plays.** `learned_raaga()` rebuilds the raaga
from stored facts, with phrases it has actually heard ahead of the ones the
library asserts, and the composer generates from that view. Studying changes
the music; a correction from the creator lowers confidence in specific phrases
and they stop being used.

**Known weakness, deliberately not hidden.** The agent's own critique of its
tunes still scores `phrase_authenticity` and `interest` low: it stays inside
the raaga and cadences correctly, but it quotes the raaga's idioms less often
than a musician would. That is the next piece of work, and the evaluator says
so on every tune rather than the number being quietly reweighted.

## Language-model providers and routing

Added 2026-09-02. The composer had one optional cloud adapter and an
all-or-nothing switch. It now has several backends and one place that chooses
between them. What the application asks for did not change.

**Five tasks, not one.** `providers/tasks.py` records what each of the five
`LLMProvider` methods actually is: how hard it is, whether its quality is
heard, whether it is on the critical path of a spoken instruction. Writing a
line to an exact syllable count and stress pattern is not the same job as
deciding that "add veena here" is an `arrange.add`, and a router that cannot
tell them apart either overpays for the second or underserves the first. The
taxonomy is data next to the interface it describes rather than knowledge
spread through the adapters.

**One prompt per task, whoever answers it.** `providers/prompts.py` holds the
wording and the parsing for all backends. A local 3B model and Claude are then
answering the same question, so moving a task between them does not silently
change what was asked, and a poor answer is the model's doing rather than the
prompt's. Parsing is deliberately forgiving - fenced blocks, a preamble
sentence, `{"lines": [...]}` instead of a bare list - because small local
models are much less obedient about returning bare JSON than a frontier model
is, and every backend needs the same tolerance.

**One model per adapter instance.** `ClaudeLLM` is one model, and the registry
builds two of them - a strong one and a cheap fast one. An adapter is not in a
position to know whether it is the right choice for a job; that is the
router's decision and it needs two candidates to make it.

**Routing on complexity, then cost, then what is actually up.** A hard task
sorts by strength and goes to the best backend running. An easy one sorts by
price and goes to the cheapest, which is a local model when there is one,
because local is free. A latency-critical task - intent classification, which
is reached mid-sentence after the rule tables have already failed - prefers a
local backend outright: an answer now beats a better answer after a network
round trip. A middling task takes the cheapest backend that clears a strength
floor, so the small local model is its fallback rather than its first choice.
`strength` and `cost_per_mtok` on the provider interface are coarse on
purpose: their job is to order four candidates, not to predict anything.

**Failure is a routing event, not an error.** A backend that raises - offline,
rate limited, refused, a model that went away - hands the task to the next in
the chain. So does one that returns nothing parseable. Beneath every chain is
the floor that was always there: no answer means the caller uses the built-in
rule and lexicon engines, which is why the application still works with
nothing installed and no key. That path is the default, not the degraded case.

**A key is noticed while running.** The registry hands the router *factories*,
not instances, so a backend can be rebuilt later. Before each request the
router re-reads whether a key exists - an environment or file lookup,
deliberately not a network call, so it stays out of the way of a spoken
instruction - and rebuilds its backends when that answer changes. Adding
`ANTHROPIC_API_KEY` or a line to `credentials.json` switches Claude on with the
application already running. No restart, no code change. That was the
requirement; this is the mechanism.

**Thinking only where it pays, and with room to do it.** Adaptive thinking is
sent on the two quality-critical tasks and nowhere else, because elsewhere it
buys latency and nothing else. Where it is on, `max_tokens` is raised well
above the task's own ceiling: the reasoning and the answer share that budget,
and a long think against a tight ceiling truncates the reply we actually
wanted. `output_config.effort` and adaptive thinking are sent only to models
that accept them - `MODELS[...].effort` is a correctness flag, not a
preference, since sending either to an older model is a 400.

**Claude Opus 5 is the default for the heavy tier**, Haiku 4.5 for the light
one, both overridable in settings without touching code. The previous default
was Sonnet 5; it remains a valid choice and is in the table.

**A capability floor on the two tasks whose quality is heard.** This one is
measured rather than reasoned. Running `llama3.2:3b` through Ollama on this
machine, CPU only: intent classification, instrument choice and a spoken
answer all came back in 6-46 seconds and were sensible. Ten lyric lines took
**704 seconds and produced nothing usable**; the run before it answered in
Tamil script, which `fitting.syllabify` cannot count and the synthesiser
cannot sing, so every line scored zero syllables. The built-in lexicon engine
fits the same ten exactly, in well under a second. A backend below the floor
is therefore excluded from `write_lyrics` and `suggest_raagas` rather than
ranked last: trying it is not a slow answer, it is a three-minute wait for a
worse one. The floor is on capability, not on being local - `llm_local_strength`
is the dial, and a genuinely larger local model clears it.

**Two things a real model exposed that stubs had not.** Ollama's JSON mode
constrains output to a JSON *object*, so a prompt asking for a bare array got
`{"raaga": ...}` where a list was expected, and `{"1": "...", "2": "..."}`
where numbered lines were. Both are reasonable readings of the request, so the
prompts now name the exact shape they want and the parsers accept the
numbered-key and single-object forms as well. Separately, a drafted line that
cannot be syllabified is now replaced by position in `lyrics/generator.py` -
that guards against *any* backend ignoring the transliteration instruction,
Claude included, and it is covered by `test_lyrics_script_regressions.py`.

**Nothing downloads a model.** Ollama and llama.cpp adapters exist and are
tested; neither fetches weights. An adapter with no model reports itself
unavailable and names the one command that would fix it, exactly as the speech
backends do. Ollama being up with the wrong model pulled is reported as
unavailable rather than answered with a substitute, for the same reason the
arranger never quietly swaps an instrument the creator named.

## Not built, deliberately

* Video, dialogue, scene generation and lip sync - specification section 24
  puts these out of scope. The timeline model is time-based and track-based, so
  adding a video track later does not require restructuring.
* Foundation-model training - section 11 rules it out for version 1.
* An installer beyond a PyInstaller bundle plus shortcut script. An MSI or
  Inno Setup wrapper is a packaging step, not an architectural one.
