# YouTube as a curriculum source

Written 2026-09-05 against the creator's four-step brief: find videos and let
me approve them; extract structured lessons into the knowledge base; convert
them into quizzes and exercises; mark the agent proficient only when it
passes. Same shape as the other plans - what is asked, what exists, the
design, the queue, the decisions, the evidence.

## The brief, and the one rule that shapes it

The creator wants YouTube used **as a curriculum source**: a place to find
what to study and to turn into lessons and quizzes. Not as a source of audio
for the composer.

That distinction is already the project's own, and it decides most of the
design. `docs/DECISIONS.md`, "The Training tab":

> **Heard and stated are different kinds of evidence.** A phrase the system
> tracked, identified and timestamped is an observation and may reach the
> composer. A phrase a teacher merely stated in a transcript has not been
> verified by ear: it is stored in the training record where a person can
> see it, and deliberately withheld from the music.

A video transcript is *stated* knowledge. So it may legitimately produce
concepts, claims, lessons and quizzes - and must not produce prayogas that
reach `generate_tune`. This is not a limitation invented for this plan; it is
the rule that already stops "we read that this is a Kambhoji phrase" from
becoming "this is a Kambhoji phrase".

The Agent Factory has the matching rule on the other side
(`factory/mastery.py`, TEST 7): without at least one graded `TestResult`,
mastery never reports above **L3 - can explain**. Stated knowledge can carry
an agent to "can restate, can explain" honestly, and no further. Getting past
L3 needs application, which needs the practice engine, which needs material
the agent has actually heard.

**So the workflow's honest ceiling is L3 per concept**, and that is the right
answer rather than a shortfall: passing a quiz about Sindhu Bhairavi is not
being able to play it.

## What exists

| Step | Requirement | State today |
|---|---|---|
| 1 | Find candidate videos | `WebLeadProvider` (`training/search.py`) exists, is off by default, records leads and fetches nothing. Its `finder` callable is `None`, so nothing populates it |
| 1 | Creator approves | Built and enforced: unticked checkboxes, "approval is never a default", persistent queue, one source at a time |
| 2 | Ingest | `training/pipeline.py` phases A to G, `training/objectives.py`, Learning Report, KB with provenance. A URL alone returns `METADATA_ONLY`; a supplied transcript is read (`training/access.py`) |
| 2 | Structured *lessons* | Missing. The pipeline extracts musical claims, not curriculum-shaped lessons with objectives, examples and common errors |
| 3 | Quizzes and exercises | `RagaTrainer.build_tests` generates the T0 to T10 ladder - but only from **shipped curriculum units** (`next_lesson` walks `curriculum.next_unit()`). Nothing turns an ingested source into a `Lesson` |
| 4 | Proficiency only on passing | Built and solid: `factory/mastery.py` evidence rules, the L3 cap, `factory/gates.py` promotion gate, `test_af07_no_false_mastery` |

The gap is narrower than it looks. Steps 1 and 4 are largely done; the
missing link is **step 2 to step 3**: turning an approved source into
framework `Lesson`s that the existing trainer can already build tests from.

## Design

### A source becomes a lesson, not a phrase

New `training/lessons.py`: `lessons_from_source(report, source) ->
List[Lesson]`. It reads what the pipeline already extracted - the learning
objectives, the claims, the report's "what I understood" - and emits
framework `Lesson` objects (`factory/models.py`) with `concept`,
`explanation`, `examples`, `common_errors` and `prerequisites`, each carrying
`KnowledgeClass.HEURISTIC` and provenance naming the video.

Every lesson is stamped `origin="stated"`. That stamp is what the rest of the
system reads to decide what a lesson may be used for.

### The trainer already knows what to do with a lesson

`RagaTrainer.build_tests(lesson, profile, history)` needs no change in
principle: it builds the ladder from a `Lesson`. What changes is where
lessons come from - `next_lesson` gains a source of stated lessons alongside
the curriculum's own, and prefers whichever the curriculum says is due.

**A stated lesson only generates T0 to T2** - recognition, recall,
explanation. Those are answerable from what a teacher said. T3 and above are
application, and an application test built from a transcript would be
grading the agent on something nobody verified by ear.

### Proficiency

Nothing to build: `apply_evidence` already caps a concept at L3 without a
graded `TestResult`, and a T0 to T2 pass *is* a graded `TestResult`, so a
stated lesson can legitimately carry a concept to L3 and stop there. The
LEARN Practice view already shows per-concept mastery, so a concept sitting
at L3 with "stated" provenance is visible without new UI.

### Where the audio question goes

Unchanged and deliberately so. A video is a lead; the creator supplies audio
they are entitled to use if they want the agent to *hear* it, and that path
already works and already feeds the composer. This plan does not touch it.

## Queue

| # | Task | Touches | Proves |
|---|---|---|---|
| Y1 | A YouTube finder for `WebLeadProvider`, off by default, behind a setting; results carry title, channel, duration, URL and `already learned` status | `training/search.py`, `core/settings.py`, tests | **done** - pasted links become leads with no key and no network; a phrase reaches the Data API only with a key *and* the web switched on |
| Y2 | `training/lessons.py`: an approved source's report becomes framework `Lesson`s stamped `stated`, with provenance | new `training/lessons.py`, `training/queue.py`, `training/controller.py`, `agent/music_agent.py`, `app.py`, tests | **done** - one lesson per concept the source actually taught, stamped `stated:`, filed to the factory store, and asserted never to reach `repo.phrases` |
| Y3 | Stated lessons reach the trainer; T0 to T2 only; LEARN shows them beside curriculum lessons | `agent/trainer.py`, `agent/student.py`, `agent/music_agent.py`, `ui/learn_workspace.py`, tests | **done** - a quiz built from a video, answered from the knowledge base, graded against what the source said |
| Y4 | The ceiling, asserted: a concept taught only from transcripts reaches L3 and stops | `tests/`, `docs/DECISIONS.md` | **done** - `test_a_concept_taught_only_from_a_transcript_stops_at_can_explain`, and a real cycle reaching exactly L3 |

Y1 and Y2 are independent. Y3 needs Y2. Y4 needs Y3.

## Provisional decisions

* **A stated lesson never becomes a prayoga.** Structural, not remembered:
  `lessons_from_source` writes to the factory store, which the composer does
  not read, and never to `repo.phrases`, which it does.
* **T0 to T2 only from stated lessons.** An application test built from a
  transcript grades the agent on something nobody heard.
* **The L3 ceiling is the framework's existing rule**, not a new one. It is
  reported rather than worked around.
* **The web provider stays off by default** and gains a setting rather than
  becoming the new normal. Nothing about the application changes for a
  creator who never turns it on.
* **Not built here:** fetching audio or video from any platform; that remains
  the creator supplying a file they are entitled to use.

## The one decision that is genuinely the creator's

**How a transcript is obtained.** Two routes, and the difference is a policy
call rather than an engineering one:

1. **The creator supplies it** - paste or file. Works today with no new
   permission, no key, and no question about terms. The finder in Y1 then
   only ever produces leads, and the creator pastes the transcript for the
   ones they want studied.
2. **The application reads YouTube's published captions** through the
   platform's own API. This needs an API key and a decision that captions a
   platform publishes are content it has handed over. It is materially more
   convenient and materially more of a commitment.

The plan above works either way and Y1 does not depend on it; Y2 does. I have
not assumed an answer.

## Hard blockers

None for Y1 and Y2 under route 1. Route 2 needs the creator's decision and a
key.

## Evidence this will be judged by

- A search phrase returns approximately ten candidates, none fetched, all
  requiring approval.
- An approved transcript produces lessons whose concepts a person recognises,
  with provenance naming the video.
- Those lessons generate answerable T0 to T2 tests, graded by the existing
  trainer.
- A concept taught only from transcripts reports L3 and not above, asserted
  by name.
- `repo.phrases` is unchanged by the whole flow, asserted by name.

## Status, 2026-09-05

Y1 and Y2 are built, on the creator's chosen route: **transcripts are
supplied by the creator, and the captions-API seam is left open** rather than
taken.

What works now, end to end: paste YouTube links into the Training search box
and each becomes a lead - no key, no network, nothing fetched.  Approve one,
supply the transcript, and the pipeline studies it as it studies any source.
When the run completes, `TrainingQueueService.on_report` hands the report to
`MusicAgent.file_stated_lessons`, which turns what the source taught into
framework `Lesson`s and files them in the factory store.

Every one of them is stamped `stated:<url>`, carries
`KnowledgeClass.HEURISTIC`, and declares T0 to T2 as its ceiling.  A test
asserts that none of it reaches `repo.phrases`, which is the table the
composer reads.

What is not built yet: Y3, which lets those lessons reach `RagaTrainer` so
quizzes are actually generated from them, and Y4, which asserts the L3
ceiling by name.  The trainer already builds a ladder from a `Lesson`; what
it does not yet do is take lessons from anywhere but the shipped curriculum.

## Status, 2026-09-05, Y3 and Y4

The loop closes.  A studied source becomes lessons; the trainer offers a new
one the moment it exists, builds T0 to T2 questions from it, and grades the
answer against what the source actually taught.  A real cycle on a
video-derived lesson ran T2, scored 1.0, passed, and took the concept from
*unknown* to **can explain** - and stopped there, which is the point.

Three decisions worth recording:

* **The agent answers from the knowledge base, never from the lesson.**  The
  lesson is what the source said and is the thing being graded against;
  answering out of it would be reading the answer back rather than showing
  what was retained.
* **A stated lesson jumps the queue exactly once, when it is new.**  The
  curriculum is the spine and carries on; what a source taught is examined
  while the creator still remembers approving it, and afterwards comes round
  through the ladder like anything else.
* **The grader is deterministic**, like every other judge here: word overlap
  between what was kept and what was said.  A provider-backed reading is what
  the escalation hook is for, not something to slip into the default path.

LEARN's mastery table gained a **Taught by** column - "a source (stated)" or
"practice" - so the difference between knowing about a raaga and being able
to play it is on screen rather than inferred from a level number.

### What the quiz does not yet prove

The agent passed by recalling a fact the shipped library had already given
it, not something the video taught.  For a raaga the library does not cover
this is a real retention test; for one it does, the agent can pass without
having learned anything from the source.  Checking that the recalled fact's
provenance is the source being examined on would close that, and is the
obvious next increment.
