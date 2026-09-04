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

**Known weakness, now measured.** The agent's own critique of its tunes
scored `phrase_authenticity` and `interest` low. On 2026-09-04 the numbers
were taken (`tools/measure_composer.py`, Keeravani taught by the reference
provider, 45-second tunes, seeds 1..20): authenticity 0.751, interest 0.437,
originality 0.775, overall 0.823. The composer now quotes fragments, treats
scale runs as facts and follows a learned idiom (see "The learning loop and
the Agent Factory" below); with all three, authenticity 0.745, interest
0.442, originality 0.733, structure 0.857, overall 0.820. Authenticity and
originality are both scored by matching the same heard phrases and pull
against each other by construction; interest is scored on distinct notes,
rhythmic variety and span, which no guidance yet touches. Both are evaluator
calibration questions, open, and the script that answers them is in the
repository rather than the number being quietly reweighted.

## Preparing real recordings

Added 2026-08-31.  The analysis pipeline was written against rendered
exercises: one voice, alone, pitched throughout.  A recording of an actual
lesson is a person talking, over a drone that never stops, occasionally
singing, and handing that to the tracker unchanged produced phrases nobody had
sung - which then went into permanent memory with provenance and a confidence
and were handed to the composer as the raaga's own idioms.  Measured against a
melody we knew note for note, 10 of the 12 phrases learned from a lesson-shaped
recording were inventions.  `agent/preprocess.py` sits before the ears.

**The drone is a gift, not a nuisance.**  A tanpura exists to declare Sa.  Where
one is found its fundamental is handed to `analyse()` as a *fixed* tonic rather
than being inferred from the melody, which is both more accurate and more
honest about where the number came from.  This is the same argument the
exercises already made for supplying the Sa when naming a single note.

**A drone is found by what does not move.**  The median over time of each STFT
bin is high where something sounds in every frame and low where a note passes
through, so the ratio of median to mean is a direct measure of how stationary a
bin is.  Candidates are then scored by summing that stationary spectrum over
the ratios a tanpura actually sounds - the Pa below, Sa, its octave and
harmonics - so the pitch that best explains the whole steady picture wins
rather than merely the loudest steady bin.

**A held note is not a drone.**  A sustained note in an alapana is stationary
too, and notching it out would remove the thing worth learning from.  Two
things separate them, and both are required: a drone sounds for essentially the
whole recording, and it brings a chord of partials where a held note brings
only itself.  Measured, held notes sit near 0.5 stationarity with one or two
partials; a real drone is above 0.89 with four.

**The tonic is refined below the bin.**  A 4096-point bin is about 5 Hz, which
near a low Sa is most of a semitone, and every swara downstream is measured
from that number.  Parabolic interpolation on the peak, combined across the
partials - a partial at four times the fundamental carries four times the
absolute error for the same relative error - takes a 45-cent error down to
about one cent.

**Singing is held pitch; speech is not.**  The gate needs no language and no
model.  A sung note settles and stays; where it is ornamented it oscillates
*around* a centre.  Speech glides from the start of a syllable to the end and
moves on.  A running median over rather more than one oscillation reduces a
kampita to the note it decorates but leaves a glide gliding, and the share of
voiced time then spent inside a plateau separates the two with a wide margin.
Gamaka swinging 140 cents still reads as singing; that was the failure this
must not have.

**Windows are two seconds, and a run must last three.**  Speech is not
uniformly unmusical: at a turning point in its contour it really does level off
for a moment, and a short window sees only that moment.  Two seconds contains
the glide either side.  A sung stretch shorter than three seconds is dropped
anyway, because it is not a phrase worth learning and an isolated flat moment
in a minute of talking is far more likely to be the speaker than a snatch of
song.

**Nothing is spliced.**  Rejected stretches are silenced in place.  Cutting
them out would move every timestamp after the cut, and the silence left behind
is a phrase boundary `segment_phrases` already knows how to read.

**Only supplied audio is prepared.**  Reference exercises and the application's
own renders are already one clean voice with nothing else in the room; running
the gate over them could only take good phrases away.  The decision is made on
the source's rights status, so it follows the material rather than the caller.

**Where it is unsure it stands down.**  If the gate rejects an entire
recording, that is far likelier to be a gate that does not suit the recording
than a lesson with no singing in it, so the audio is left alone and the warning
says so.  A phrase never learned costs one phrase; a wrong phrase in permanent
memory costs the confidence of everything that reads it.

## The Training tab

Added 2026-08-31 from the Training tab / autonomous learning specification.
The creator searches for material, chooses what may be learned from, and gets
a report saying what was understood, what was learned, and where every fact
came from.

**A second store, not more tables in `knowledge.db`.** The two answer different
questions - one holds what the agent knows about music, the other the record of
how training was conducted - and keeping them apart means a training history
can be thrown away without touching what the agent learned from the creator's
own recordings. The link between them is the provenance on each knowledge
entry, which names the source and the run.

**Search is providers behind one normalised result.** The tab never learns what
a provider is; it asks for candidates and gets `LearningSource` objects. Three
ship. `exercises` renders material from the shipped raaga library and listens
to its own playing - always available, no network, no rights question, and the
reason the whole loop works on a machine with nothing else. `library` searches
the creator's own learning folder. `web` is off by default and records *leads*
only.

**Ranking is by usefulness for learning, not textual relevance.** A source
whose content can actually be analysed outranks one we can only name, because
the second cannot teach anybody anything. A perfectly-titled lead therefore
loses its place to an exercise, which is the correct answer to "which of these
will teach the system something".

**Raaga names are matched across transliterations.** The specification's own
acceptance test asks for "Kamboji"; the library calls it "Kambhoji". Matched
literally, the one search the specification names returned nothing. Aspirates,
doubled vowels and the v/w and i/y pairs are normalised away for matching only -
never for renaming.

**Nothing is fetched.** `access.py` is the single gate, and there is no code
anywhere in the feature that logs in, follows a paywall, strips DRM or
downloads from a platform that has not handed the file over. A source on the
network is marked metadata-only and its report says
`METADATA ONLY - CONTENT NOT ANALYZED` in the specification's own words, with
the two honest ways forward offered: supply the file, or supply a transcript.
Both work, and turning a lead into a real lesson that way is tested end to end.

**Video files are refused with a reason.** There is no demuxer in the
application, so an `.mp4` is reported as needing its audio extracted rather
than failing three phases later with something that reads like a bug.

**Heard and stated are different kinds of evidence.** A phrase the system
tracked, identified and timestamped is an observation and may reach the
composer. A phrase a teacher merely stated in a transcript has not been
verified by ear: it is stored in the training record where a person can see it,
and deliberately withheld from the music. Storing both and playing only one is
what keeps "we read that this is a Kambhoji phrase" from becoming "this is a
Kambhoji phrase".

**Uncertain inference does not become fact.** Below a confidence floor an
observation is reported as uncertain in the report and left out of the
knowledge base, rather than written down slightly hedged where it will later
look authoritative.

**A conflict is never resolved by overwriting.** The existing claim stays
exactly as it was and is flagged; the new one is kept beside it with its
evidence, and a recommendation is offered to a person. Two teachers disagreeing
is a fact about the material, not a bug to be tidied away.

**Approval is never a default.** Search results arrive with an unticked
checkbox. A pre-ticked box is not approval, and section 20 rule 1 makes the
choice the creator's alone. Autonomous search does not mean autonomous
approval: the tab will suggest a phrase from a curriculum gap, and still queue
nothing until somebody picks it.

**The queue is the `runs` table.** There is no separate in-memory queue to keep
in step with the database, which is what makes surviving a close trivial rather
than careful. A run left mid-flight by a crash is in a working status with
nobody working on it, so on startup those are returned to Queued with their
attempt count intact rather than sitting in "Analyzing" for ever.

**Every completed source has a report, including the ones that failed.** A
source that taught us nothing still has to say so and why; rule 4 has no
exception for failure.

## The Knowledge Base

Added 2026-08-31 from the knowledge-base architecture specification.  Its
section 47 asks for the existing project to be inspected first, and that
inspection changed the design, so it is recorded here before anything else.

**What was already here.**  Three stores.  `agent/knowledge.db` (sources,
phrases, raaga facts, curriculum progress, compositions, feedback, events);
`training.db` (searches, candidates, runs, objectives, reports, a flat
knowledge table, conflicts); and `raaga/data/raagas.json`, the shipped
structural library.  The specification forbids a parallel Knowledge Base, so
none of them was duplicated.

**One node table, not four.**  Section 4 defines a Knowledge Item with a
statement, section 6 a Claim with subject, predicate and object, and section
33 lists `knowledge_items`, `entities`, `claims`, `procedures` and `examples`
as separate tables.  Implemented literally that is four identity spaces
holding overlapping content, and a relationship - whose endpoints are
`knowledge_id` - could not point at three of them.  So there is one node table
and one id space: a claim is an item with its subject and predicate filled in,
an entity is an item of an entity type, and procedures and examples have a 1:1
detail row for the extra fields they need.  The specification's table names
survive as read-only **views**, so the logical model it asks for can be
queried by those names while exactly one row holds each piece of knowledge.

**Where each existing store went.**  `training.db`'s `knowledge` and
`conflicts` are migrated into the Knowledge Base.  Everything else in that
file stays, because section 26 draws exactly that line: a Learning Report is
what happened during one run, the Knowledge Base is what accumulates across
all of them.  The agent's raaga facts are projected in as claims while its own
store keeps them too - the phrase index is on the composer's hot path and is
already proven, and curriculum progress is run state rather than knowledge.
The shipped library is seeded as claims attributed to a source naming the file
they came from, so that a teacher later disagreeing with one produces an
honest conflict between "what we shipped" and "what a source taught" rather
than a mystery.

**Opening is not creating.**  Section 2 is the rule the store is shaped
around.  A durable marker records the moment of first initialization and every
later open continues from it; initialization and migration are separate code
paths so a migration cannot fall through into a recreation; a store written by
a newer schema is refused outright rather than misread; and asking for a store
that is not there, without asking to create one, is an error rather than a
fresh empty database that looks like total loss.

**A damaged Knowledge Base is kept.**  Section 36.  Corruption stops
destructive writes, preserves a timestamped copy and reports - it is never
replaced with an empty one.  That covers both a file that opens but fails its
integrity check and one too damaged for SQLite to open at all; the second case
was a defect found by its own test.

**Set-valued predicates.**  A raga has one arohanam, so a second different one
is a contradiction.  A raga has many characteristic phrases, so the second is
simply another one.  Identity therefore includes the value for set-valued
predicates and not for single-valued ones.  Without that distinction every
phrase after the first was recorded as contradicting its predecessor, seeding
the library produced a conflict per prayoga, and the store grew on every
start.

**Refinement compares substance, not sentences.**  Two arohanams differing by
one swara read as very similar text.  A refinement is a better *wording* of
the same claim, so it requires the structured values to agree; where they
differ it is a contradiction however alike the sentences look.  Getting this
wrong was a silent overwrite, which is the thing section 12 exists to prevent.

**Claims hang off the thing they are about.**  Section 3's core idea.
Committing a claim about a raga brings that raga into being as an entity if it
is not already there and links the claim to it, which is what makes graph
traversal from "Kambhoji" reach its phrases and its constraints.  An entity
asserts only that a name denotes a thing of a kind: it carries no evidence,
because nobody taught it, and asking where it came from says so rather than
returning a blank that would read like lost provenance.

**Confidence keeps its working.**  Section 10 lists eight considerations and
section 41 asks why a thing is believed.  A single number computed and stored
alone cannot answer that, so the components are stored beside it and a score
can be read back as a sentence.  Independence is counted rather than evidence:
ten records from one video are one source agreeing with itself.

**Semantic search is declared missing rather than faked.**  Section 18 lists
it among the hybrid routes and section 32 makes embeddings an optional later
addition.  There is no embedding model here, so `semantic_search` says so and
falls back to keyword matching, and `semantic_available` is False.  A lexical
match dressed up under that name would be a lie about how an answer was found.

**Retrieval is ranked for usefulness and never hides a disagreement.**  A
context carries the constraints whatever the task profile asks for, and where
sources conflict the caller is told - rather than being handed the more
confident of two answers as though it were settled.
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

## Canonical specification v0.3

Added 2026-09-03. The four documents the sections above were written against
are consolidated into one, `docs/spec/CANONICAL_SPEC_v0.3.txt`, with its own
numbering. `docs/spec/CROSSREF.md` maps every old section number cited in the
code and in this file to its v0.3 home; `docs/PLAN_v0.3.md` is the work queue
and status report. The decisions above stand unless a line here says
otherwise.

**The specification text is kept verbatim, not rewritten.** A section number
quoted in a commit, a test name or a conversation must resolve to the
creator's own wording, and a Markdown rewrite would drift from it within a
week. Cross-referencing costs one table; re-numbering sixty-seven comments
costs a diff with no behaviour in it.

**Learning leaves the composer's tab bar.** Section 4 makes MAIN and LEARN
two top-level workspaces, and TEST I fails the previous layout by name. The
Learning and Training tabs were built as siblings of Tune and Lyrics because
the earlier specifications asked for tabs; the widgets inside them are kept
and re-homed under LEARN's six areas (dashboard, curriculum, sources,
practice, knowledge, history). The two workspaces are pages of one window
rather than two windows, so the conversation dock, the status bar and the
transport keep one owner.

**Apply Brief does the whole job.** Section 6 lists eleven things the button
must do; before this it did one, saving the fields, and the suggestion lived
behind a second button. That read as "nothing happened", which section 6.1
forbids. The second button stays as the way to ask for alternatives (section
7); both run one action that reports its phases.

**Every important action reports a state, not just a message.** Section 6.1's
Idle / Starting / Working / Completed / Failed / Cancelled contract is one
small model (`core/actions.py`) that the status bar, the panels and the
project's error log all read from. It is adopted by Apply Brief first because
that is the one the specification names; other actions move to it as they are
touched.

**Secure key storage is `keyring` where present, the file where not.**
Section 42 says "OS-appropriate secure credential storage where practical".
On Windows `keyring` writes to Credential Manager and installs without a
compiler, so it is an optional dependency rather than a core one: the
application must still start with only `requirements.txt`. The environment
variable keeps precedence so `setx` still works and a build machine can
inject a key without a UI. The suite forces the file backend so no test can
touch the creator's real credential store.

**Provider status is data first, a table second.** Section 41 wants Configured
/ Not configured / Unavailable and Available / Not installed / Ready visible.
The registry already knew all of that in the form of log lines; it now
returns it as `ProviderStatus` rows, and the settings dialog, the diagnostics
bundle and the LEARN dashboard render the same rows rather than each asking
the backends again.

**Secrets are redacted at the formatter.** A key that reaches a log line
through an exception message or a request dump is masked before any handler
writes it, and the diagnostics export scrubs the same patterns again and
never bundles `credentials.json`. Two layers because the second exists to
catch what the first was not written for.

## The learning loop and the Agent Factory

Added 2026-09-04. Two plans, `docs/PLAN_learning_loop.md` and
`docs/PLAN_agent_factory.md`, and eight pull requests (#3 to #10). The
decisions above stand unless a line here says otherwise.

**A retry is a fresh attempt, seeded from the unit and the attempt number.**
REG-095 failed in three of four full-suite runs because the practice seed was
the wall-clock second: every retry made within one second replayed the failed
attempt note for note until the twelve-attempt cap. `practice_seed(unit,
attempt)` is a CRC of both, so the same attempt sets the same exercises in
every process and every retry sets different ones. The hash-seed suspect was
real but was not the cause; freezing the clock in a script was what
reproduced it.

**A failure is knowledge, not a log line.** The evaluator's mistakes are
structured findings; a failed attempt writes lessons with the specification's
Failure/Lesson fields (v0.3 section 38); a recurring mistake is one row that
counts, not a duplicate. Guidance built from those lessons constrains the
next attempt without telling the evaluator, so a pass still measures the
agent. Guided retries fixed 16 of 20 seeds that failed the short-phrase unit
on originality; the remaining four are the originality checker penalising
scale runs in six-note lines.

**A scale is not a phrase.** The reference provider renders the arohanam and
avarohanam; research stored them as eight-note phrases at confidence 0.97,
and they became the composer's favourite quotations and the loudest idiom
evidence. A monotone run of six or more notes is treated as the fact it is:
kept in the bank the evaluator reads, never a prayoga, never idiom.

**The idiom is a prior-weighted shading, attached to the learned view only.**
`RaagaIdiom` adds heard moves to the composer's own habits expressed as four
pseudo-observations, so a handful of phrases shades choices rather than
replacing them. An unstudied raaga has no idiom and composes byte for byte as
before; the golden melodies are the guard.

**The learning framework is a package that knows no music.** The Universal
Learning Framework v0.1 (`docs/spec/agent_factory/`) lives in
`raagacomposer/factory/`: ladders L0 to L9 and T0 to T10, four data splits,
an adaptive trainer, a Judge that is a function building a throwaway object,
the ten-step cycle, promotion and release gates, and a cumulative maturity
ladder. A toy plural-rules domain proves it without music and carries the
framework's acceptance tests in domain-free form. The Raga agent is the first
Student; `RagaTrainer` and the library's hard rules are its Trainer and Judge.

**Hard knowledge is the library; heard knowledge is heuristic.** A learned
fact that contradicts the library loses a dispute to a hard rule, the ruling's
correction goes in at full confidence, and the claims it contradicts are
overruled rather than left to share a key. The learned view keeps the most
trusted claim per key; it used to keep whichever the store returned last.

**Running the loop for real is part of verification.** Every increment's
implementation arrived with green unit tests, and a probe that ran the real
loop found seam defects the tests could not: a curriculum never told a unit
passed, a ladder with no higher rung to promote to, disputes raised on passed
results, a rule ruling on the word "invalid", feedback that carried no
actionable kinds, praise that jumped a concept from L0 to L9. Each probe is
now a regression test.

**The creator is the external evidence.** Rejecting a tune opens a dispute
between the agent's verdict and the creator's, resolved at once in the
creator's favour with a candidate reusable lesson; the creator's words are
mapped by a small lexicon into the evaluator's finding kinds so they become
guidance for the next tune. Praise is field evidence (L9) only once the
capability has reached L7 on real-world passes: one success is not a rule.

## Stage 1 knowledge pack

Added 2026-09-04. `docs/spec/stage1_knowledge_pack/` holds the creator's
Stage 1 pack verbatim: the swarasthana dictionary, all 72 melakarta scales
with block-character heuristics, the brief-to-raga selection engine and its
acceptance tests. It is treated as guidelines for what the agent should know
at stage 1, not as a rewrite of the library.

**Its [GRAMMAR] is hard knowledge, its [HEURISTIC] is heuristic.** That is
the same distinction the Agent Factory already draws (`KnowledgeClass`), so
the pack slots in without a new concept. `tests/unit/test_stage1_pack.py`
runs the pack's own validation rules and mandatory unit tests A to C against
the files, and checks that every melakarta the library carries agrees with
the pack's arohanam and avarohanam. The heuristic tags are checked for
presence, never for truth.

**The pack is generated into library data, never read at runtime.**
`docs/spec/` is frozen so that a section quoted in a commit resolves to the
creator's own wording, and a runtime that reads its text would make it a
dependency instead of a document. `tools/build_melakartas.py` parses the
three map files, runs the pack's own validation rules (document 06) and
writes `raagacomposer/raaga/data/melakartas.json`, which the library loads.
A unit test regenerates and compares, so the two cannot drift apart in
silence; `--check` is the same comparison from the command line.

**A melakarta is matched by its number, never by its name.** The library
calls melakarta 65 Kalyani and the pack calls it Mechakalyani, so the merge
needs some way to see that they are one raaga - but names are the wrong
instrument for it. "Bhairavi" is a janya and "Natabhairavi" is melakarta 20,
and no amount of transliteration-matching separates those two safely. The
melakarta number is what both sides actually assert, so it is what they are
joined on; the names are then checked against each other and a disagreement
is logged rather than guessed at.

**A curated entry wins, and gains only what the pack knows.** The eight
melakartas already in `raagas.json` carry prayogas, jeeva swaras, resting
notes, gamaka and a tempo range that the pack does not have; they keep every
one of them and gain the chakra, the three blocks, the block characters and
the starter tags. The pack's spelling becomes an alias, so one melakarta is
one entry (sections 34 and 35).

**A melakarta nobody curated is a scale, and says so.** The other 64 join
with their arohanam, avarohanam and block character and nothing else - no
invented jeeva swaras, no resting notes, and no tempo range, because section
37 ends "unknown fields remain unknown; do not fabricate completeness". They
compose from the scale alone and `describe()` and `character()` state the
limit rather than implying a depth that is not there. `Raaga.scale_only` is
that condition, and it stops being true the moment the agent hears a phrase.

**A raaga with no curated character does not answer a thin brief.** An empty
brief used to hand every raaga the same small bonus, which with 72 melakartas
would have filled the list with scales nobody could give a reason for. The
bonus now goes to raagas that have something curated to say; the pack's
block-character scorer (`docs/PLAN_stage1_knowledge.md` S2) is what will
speak for the rest. Asked for by name, a scale-only melakarta is selected,
composed from and explained exactly like any other.

**Seventy-two names in one namespace need whole words.** Bare substring
matching was safe with eighteen raagas. With seventy-two, a name buried
inside an ordinary word starts answering briefs nobody wrote, so
`find_in_text` matches whole words and still prefers the longest name in the
text - Mechakalyani over Kalyani, Natabhairavi over Bhairavi. `get`'s
partial-match fallback picks the longest name found inside the query, or
failing that the shortest name the query is found inside, rather than
whichever the dictionary happened to yield first. An alias claimed twice is
logged and left with its first owner.

**A bare scale does not displace a raaga a student is taught to tell apart.**
The seam a probe found rather than a test: two places rank "which other
raaga explains these notes" - the practice engine's neighbour-drift
exercises and the evaluator's drift score - and both broke ties by whatever
order the library happened to yield. With eighteen raagas that was harmless.
With seventy-two, Kalyani's four confusable neighbours went from
Shankarabharanam, Hamsadhwani, Kambhoji and Mohanam to four melakartas
picked out of dozens tied on overlap, so the exercise stopped training
against the raagas anyone confuses. Both now break ties towards a curated
raaga and then by name. A melakarta that genuinely shares more swaras is
still the nearer neighbour and still wins, so the drift exercises did get
harder - that is the library being more complete, not a regression. The
evaluator's tie-break also compared each candidate's size against the target
raaga rather than against the current best, which meant every tied candidate
smaller than the target replaced the last one; candidates are now compared
with each other. Measured: for tunes that stay inside their own raaga the
drift score is unchanged to three decimals across five raagas and twelve
seeds each, because nothing can explain those notes better than the raaga
itself.

**The brief becomes fourteen numbers, and the raaga is scored against them.**
S2, `raaga/emotion.py`. The brief's fields are read into the pack's fourteen
emotion dimensions and blended by the pack's own field weights (mood and feel
0.30 each, situation 0.25, title 0.10, notes 0.05, renormalised over the
fields that actually say something); every raaga gets a profile built from
its block characters, its starter tags and any curated moods; the two are
compared by cosine similarity, which measures the *shape* of what was asked
for rather than how loudly it was said. The pack's contradiction penalties
and block bonuses then adjust that, and the result is a score out of 100.

**One vocabulary for both sides.** A creator's "lonely" and the pack's
"plaintive" have to land in the same space or there is nothing to compare, so
one lexicon covers three vocabularies at once: the words creators type, the
block characters in pack documents 01 and 02 to 04, and the "good for" uses
those files list. A test asserts every block character in the pack is
readable by it, because a block nobody can read is a melakarta profiled from
silence.

**Negation suppresses; it does not invert.** "not sad, just tired" removes
sadness from the target and asserts nothing in its place. Guessing which of
the other thirteen dimensions the creator meant would be answering a brief
they did not write.

**The fit decides; the pack's rules tune.** Bonuses are capped at 0.15 and
penalties at 0.25 against a cosine of at most 1.0, so a raaga the brief does
not resemble cannot be carried into the list by block rules alone. The same
argument caps what the agent's own study can add (`LEARNED_BONUS_CAP`, ten
points out of a hundred): studying a raaga should move it up the list, not
carry one the brief does not resemble to the top of it. The creator asked for
a feeling, and "I happen to know this one well" is not an answer to that.

**Scores are scaled by the best a raaga could do, not clamped at it.**
Clamping the total into 0..1 put every good answer on 100 and threw away
exactly the differences the ranking exists to express - and with the scores
flat, the diversity step had nothing to trade against.

**Diversity chooses which five, not what order they come in.** Each place
after the first goes to the raaga with the best score once its likeness to
what is already chosen is discounted, so the list becomes the pack's spread -
the closest fit, then a warmer, darker, brighter or stranger alternative, or
the other madhyama - without any of those roles being hard-coded as a slot to
fill. The order stays by score: a list numbered one to five whose scores do
not descend reads as a defect, and the agent's own acceptance test requires
descending scores.

**Score says how well it fits; confidence says how much is known.** A
melakarta the pack supplied and nobody curated can fit a brief perfectly and
is ranked accordingly, but it is offered at lower confidence
(`SCALE_ONLY_CONFIDENCE`), because all there is behind it is a parent scale
and a starter tag.

**Ranking a raaga and choosing one to compose in are different questions.**
Apply Brief ranks by emotional fit and says so. `require_raaga` - which picks
only when the creator has not, with a tune about to be written - prefers,
among the raagas that fit, one there is something to compose *with*: one the
agent has studied, else one somebody curated prayogas, resting notes and
gamaka for, and a bare parent scale last. Without that split the application
answered a default brief by picking a raaga it knew nothing about and
composing in it, over one it had studied for thirty units.

**A model may gloss the reason; it may never replace it.** The score and the
sentence that explains it are derived from the block map, which is the whole
reason the pack's block model exists - a creator can check the derivation
against the map. Where a language-model adviser is configured it now appends
an attributed sentence beside that derivation rather than overwriting it, so
it stays visible which half of a claim came from where.

**A preference is learned for a feeling, not for a raaga.** S3, pack document
05 section 6. The creator's choices are stored in `selection_weights` as
`(raaga, dimension, weight)` - one row per emotion dimension the brief was
asking for, plus a `"*"` row for the raaga however it was asked for. So
rejecting Keeravani for a grieving brief lowers Keeravani *when something
grieving is asked for* and leaves it exactly where it was for a wedding. A
single scalar per raaga would have been simpler and would have taught the
agent the wrong lesson.

**Feedback is a table of its own, and that is the enforcement.** The pack
says "save feedback separately from hard grammar; do not rewrite
Arohanam/Avarohanam from preference feedback". `selection_weights` is a
separate table written by one method that has no way to reach `raaga_facts`,
`phrases` or the library, and read by nothing except the ranking. The rule is
structural rather than remembered, and a test asserts it at the level a
creator would notice.

**The pack's own signal strengths, plus one it does not name.** Accepted
+1.0, auditioned +0.2, rejected -0.7, and *passed over* -0.25: choosing the
third suggestion says something about the first two, but not as much as
saying no does. Measured on the sad/romantic/lonely/warm brief, an explicit
rejection moves a raaga about 8 points of 100 against a cap of 9.1, and being
passed over about 2.

**Evidence accumulates with diminishing returns.** The bias is
`tanh` of the accumulated weight against the brief, so more evidence keeps
moving the ranking but can never exceed `MAX_FEEDBACK`, and the stored weight
saturates at `WEIGHT_LIMIT` besides. One explicit rejection already spends
most of the budget, which is deliberate: a creator who says "no, too sad"
should see the list change now, not after the third time they say it.

**A correction is read through the same lexicon as everything else.** Rather
than a table of the pack's five examples, the modifier is read and the word
after it goes through the emotion lexicon, so "less tense" and "a bit more
mysterious" work without being enumerated, and "not warm enough" is read as
asking for more warmth rather than less. A comment that is not a correction
at all - "I like it" - moves nothing, because understanding nothing is better
than inventing a number.

**The application choosing for itself is not a preference.** `select_raaga`
takes `by_creator`, and `require_raaga`'s automatic pick passes `False`.
Counting it would have the agent learning its own habits back from itself and
calling the result the creator's taste.

**Only a choice among the suggestions we offered is feedback about a
feeling.** Found by running it: the first version attached a choice to
whatever brief was in the panel at the time, so applying a second brief and
then picking from the first list taught the agent that a raaga chosen for a
grieving brief suits a wedding - the live pass printed "prefers
Kharaharapriya for joy" after a choice made for "love failure". Attaching it
to the brief the suggestions were made for is not enough either, because
that has moved on too. The honest rule is narrower: learn only when the
chosen raaga is in the current suggestion list, because that is the one case
where the raaga and the brief are known to belong together. Naming a raaga
that is not in the list is an override, and an override is not a preference -
the selection still happens, only the learning stands down.

**Saying no needed a control.** Acceptance was already wired to "Use this
raaga"; without a matching one for rejection the creator could only ever
teach the agent by agreeing with it. The raaga panel has "Not this one",
which records the rejection and immediately re-ranks so the effect of what
they just said is visible rather than waiting for the next Apply Brief.

**Withdrawing a preference deprecates it rather than deleting it**
(framework document 04 section 6). A preference held and withdrawn is not the
same as one never held, and a creator looking at why a ranking changed should
be able to see which happened.

**Still to come:** arohanam/avarohanam audition
(`docs/PLAN_stage1_knowledge.md` S4, with pack test E). `MusicAgent.audition_raaga`
exists and sends the pack's +0.2, and nothing calls it yet.

## Local-first routing

Added 2026-09-04, replacing the complexity-tier routing above where the two
disagree. The standing policy: attempt a local model first for every
model-driven step, judge what it produced, and reach a paid model only on a
judged failure.

**The strength floor is gone from the judged modes.** It excluded weak
backends from lyrics and raaga suggestion outright, on a measurement taken
once - `llama3.2:3b`, 704 seconds, nothing usable. That froze a judgment that
hardware and models both move past, and it meant "prefer local" was quietly
not in force for exactly the two tasks where it mattered. Nothing is now
excluded before it has been tried; the judge decides on what actually came
back. The floor stays in `auto` and the `*_first` modes, where there is no
judge and a strength number is all there is to go on.

**Three signals, in order, stopping at the first verdict.** Schema validity
first, because it is free and certain. Then the mean token log-probability
where the runtime exposes one - Ollama does, llama.cpp here does not, and a
runtime that cannot say is not failed for staying silent. Then, only when
log-probabilities are missing or borderline, a second sample at non-zero
temperature, because it costs a whole extra generation.

**A deadline is a signal too.** The policy as stated judges quality, not
time, and a local model that takes ten minutes to produce a good answer has
still failed the creator. `routing_attempt_seconds` makes the 704-second case
an escalation rather than a wait.

**Numbers are compared as numbers.** The two-sample check first used text
similarity, which is nearly blind to the disagreement that matters: two
fourteen-dimension affect vectors contradicting each other on every value are
textually almost identical, because the keys and the punctuation are most of
the string. For a mapping of numbers the measure is now the mean absolute
difference across the union of keys.

**The judge earns its first signal immediately.** Asked for raaga
suggestions, `qwen3:4b` answered at a confident -0.53 mean log-probability
with one of its three entries keyed `": "` instead of `"raaga"`. Only a
schema check catches that, which is why every capability now passes a
validator and why schema is checked before anything else.

**The whole policy is one config block**, so changing it needs no code edit
and no redeploy: the mode (`llm_routing`), the models by tier
(`routing_tiers`, named rather than ordered so the loop can ask for the one
it wants), the two escalation orders - prose, and schema-constrained, which
starts at the model chosen for structured output - and the thresholds.
`claude_only` leaves the local candidates out of the chain rather than
running them and discarding the result, so a rollback costs nothing in
latency, and every mode returns the same shape so nothing downstream cares
which model ran.

**Every attempt is logged, and the mode and model are recorded with every
result.** `routing_attempts.jsonl` gets the brief, each attempt's verdict and
failing signal, what each backend actually said, and which one answered.
Without it a threshold is a guess that cannot be improved, and a quality dip
cannot be told apart from a change we made ourselves.

**The thresholds are unvalidated.** The log exists precisely because they
are: the first real measurement already contradicted one of them, since the
affect vector under an enforced schema came back at a mean of -1.73 against a
floor of -1.10, and that was a *good* answer. Numeric tokens are legitimately
uncertain - choosing 0.8 over 0.9 is not a mistake - so the floor has to be
calibrated against the log rather than reasoned about.

**`local_first` is the default, not merely available.** The point of the
policy is that a local model is tried by default rather than on request, and
it is safe to default to only because the judge exists to catch a bad local
answer. `auto` keeps the older behaviour for anyone who wants it, and an
existing `settings.json` that names a mode still wins - the default only
decides what a machine with no stated preference does.

**The configured escalation order decides the local chain, not strength.**
Registering the tiers was not enough to make them mean anything: `chain()`
ordered local backends by strength, and since every local model costs
nothing the cost key could not separate them, so the largest and slowest went
first. That is the opposite of "a cheap first attempt", and it made
`routing_order` decorative - `qwen3:8b` led every chain while `qwen3:4b`, the
tier chosen to be tried first, came last. In the judged modes the rung
decides; a local backend the config does not name sorts after the ones it
does, and a paid backend after every local one. The unjudged modes keep the
strength ordering, because there is no judge there and strength is the only
signal they have.

**One tag of a family does not stand in for another.** The Ollama probe
compared only the part before the colon, so `qwen3:8b` reported itself ready
because `qwen3:4b` had been pulled, then answered every request with a 404.
Harmless while one local model was ever configured; wrong the moment the
tiers put two tags of one family side by side. A bare name still means
`:latest` on both sides.

## Not built, deliberately

* Video, dialogue, scene generation and lip sync - specification section 24
  puts these out of scope. The timeline model is time-based and track-based, so
  adding a video track later does not require restructuring.
* Foundation-model training - section 11 rules it out for version 1.
* An installer beyond a PyInstaller bundle plus shortcut script. An MSI or
  Inno Setup wrapper is a packaging step, not an architectural one.
