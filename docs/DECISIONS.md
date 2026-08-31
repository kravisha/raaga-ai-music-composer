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

## Not built, deliberately

* Video, dialogue, scene generation and lip sync - specification section 24
  puts these out of scope. The timeline model is time-based and track-based, so
  adding a video track later does not require restructuring.
* Foundation-model training - section 11 rules it out for version 1.
* An installer beyond a PyInstaller bundle plus shortcut script. An MSI or
  Inno Setup wrapper is a packaging step, not an architectural one.
