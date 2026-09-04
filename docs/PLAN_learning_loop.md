# Closing the learning loop: retries that learn, composition that uses what was learned

Written 2026-09-04 against `docs/spec/CANONICAL_SPEC_v0.3.txt`. Same shape as
`docs/PLAN_v0.3.md`: what the specification asks, what exists, the queue, the
provisional decisions, and how each step is verified. Nothing here is built
yet.

## Where the agent stands

After PR #3 and PR #4 the loop is reliable: every retry is a fresh, seeded
attempt, and a fresh agent completes all 35 Keeravani units. Two things stop
"completes the curriculum" from meaning "gets better":

1. **A failed attempt changes nothing.** The evaluator produces twelve
   dimension scores, a list of mistakes and a recommendation (section 26),
   but the mistakes are prose strings and the recommendation is appended to a
   log line in `MusicAgent._practice_step`. The next attempt is a different
   roll of the dice with no memory of the last one. Section 24 step 6 says
   "retry weak attempts"; section 38 says "failures should prevent repeated
   rediscovery of the same mistake"; section 28 lists "negative/failure
   knowledge" as a required knowledge type. None of that exists.

2. **Learned knowledge reaches the composer as quotations only.** This
   corrects an earlier statement that it did not reach the composer at all.
   `App.composing_raaga()` asks `learned_raaga()` for the raaga as the agent
   knows it, which puts heard phrases (confidence >= 0.4) ahead of the
   library's prayogas, and `melody._phrase_tokens` quotes those prayogas
   verbatim. So studying does change the tune. But two mechanisms fight:
   the composer quotes a whole learned phrase, then `generate_tune` runs the
   originality check against the same learned phrases and regenerates blind
   (`seed += 7919`, three tries) when the quoted run is longer than the
   checker's limit. And the composer learns nothing but the phrases
   themselves: no ascent/descent tendencies, no cadence behaviour, no
   phrase-length habit (section 37 "phrase tendencies, ascent/descent
   behavior, cadence behavior"), and no lesson from the critic or the creator
   (section 26 "user feedback has high weight"; section 27 USER FEEDBACK ->
   UPDATED KNOWLEDGE / EXPERIENCE). `DECISIONS.md` already records the
   symptom as a known weakness: the agent's own critique scores
   `phrase_authenticity` and `interest` low on every tune.

## What the specification asks

| Section | Requirement | State today |
|---|---|---|
| 14 | "retries weak areas", "uses learned knowledge in composition", "learns from user critique" | retries are uninformed; composition quotes phrases; critique adjusts phrase confidence only |
| 15 #9, #11 | Retrieval / Context Builder; Evaluator / Teacher / Critic | evaluator exists; no context builder feeds practice or composition |
| 24 steps 5, 6 | Evaluate, retry weak attempts | evaluate yes; retry is a reseed |
| 26 | Evaluator output: dimension scores, detected mistakes, recommended correction, pass/retry decision | scores yes; mistakes are strings; correction is one sentence; nothing consumes either |
| 27 | PRACTICE -> EVALUATION -> RETRY / ADVANCE -> ... -> UPDATED KNOWLEDGE / EXPERIENCE | the arrow from EVALUATION back into KNOWLEDGE is missing |
| 28, 38 | Negative/failure knowledge; Failure/Lesson object (task, attempted method, result, failure reason, correction, related knowledge, source/run, confidence, date); failures prevent rediscovery | no lessons table, no use |
| 37 | Raga knowledge includes phrase tendencies, ascent/descent behaviour, cadence behaviour, common mistakes | phrases and facts only |
| 4.2 D, E, F | Practice area shows retry/remediation; Knowledge shows gaps and corrections; History shows evaluation results and composition learning history | practice runner shows scores; no remediation, no lesson history |
| 16 | "Why did you choose this phrase?" | the tune records no provenance |
| 61 | CORRECT: evaluator calibration, characteristic phrase accuracy | authenticity and interest known low |
| 64 steps 16, 17 | Connect Practice -> Evaluator -> mastery/retry; connect learned KB knowledge back into Generate Tune | both half-connected |
| 58 items 26, 27 | Evaluator can score it; Generate Tune can use learned raga knowledge | yes; partly |

## Design

### Part 1: a failed attempt teaches the next one

**Findings, not sentences.** `Evaluation` keeps `mistakes: List[str]` for
every existing reader and gains `findings: List[Finding]`, one per detected
mistake, with `dimension`, `kind`, `evidence` (the tokens or transition
involved) and `weight`. Kinds, each raised by the scorer that already detects
it: `outside_swara`, `wrong_direction` (the offending transition),
`too_many_leaps`, `no_cadence` (the actual ending), `neighbour_drift` (which
neighbour), `not_original` (the learned phrase id and run length),
`repetitive`, `no_gamaka`, `off_beat`, `no_idiom`. The report text is
unchanged, so the critic still "never collapses its scores".

**A lesson is knowledge** (section 38). New `lessons` table in
`KnowledgeRepository`, with the spec's fields: `id, at, raaga, unit_id,
attempt, task, method, result, failure_reason (kind + evidence), correction,
related (phrase ids / tokens), source_run, confidence, recurrences,
applied`. `MusicAgent._practice_step` writes one lesson per finding on a
failed attempt. `App.critique_tune` and `App.give_feedback` write lessons
against compositions, so a creator's "this does not sound like Keeravani"
becomes the same kind of record as a failed practice (section 26 examples).
Lessons are shown under LEARN: Practice / Quiz gets a "Retry / remediation"
list for the current unit, Knowledge gets "knowledge gaps" from recurring
lessons, History gets evaluation results with their findings (section 4.2).

**A context builder turns lessons into guidance** (section 15 #9). New
`agent/guidance.py`: `build_guidance(repo, raaga, unit=None) ->
Guidance`, deterministic from stored lessons. `Guidance` is a small set of
constraints, not a prompt: `avoid_transitions`, `avoid_endings`,
`must_end_on_nyasa`, `avoid_quoting` (phrase ids), `prefer_step` (leap
budget), `quote_more` (raise idiom quoting), `vary_more`, `add_gamaka`. Each
weight rises with a lesson's `recurrences`: a mistake made twice is avoided
harder than one made once. Guidance is per raaga, and the unit's own lessons
count double.

**Generation obeys guidance.** `PracticeEngine._motif` and
`_notes_from_tokens` take an optional `Guidance`. A step that would produce an
avoided transition is re-drawn (bounded); an avoided ending is replaced by a
nyasa; `avoid_quoting` removes those phrases from the bank for this attempt;
`quote_more` and `prefer_step` shift the existing probabilities. The
evaluator is never told about guidance, so a pass still means the line
passed on its own. With `Guidance()` empty, the code path and the random
draws are identical to today, so every existing expectation holds.

**Rediscovery is prevented, and visible.** When a finding of the same kind
recurs for the same unit, `recurrences` increments instead of a duplicate
row; the practice report says "second time: <kind>" in its detail. This is
the measurable version of section 38's sentence.

### Part 2: composition uses what was learned, beyond quotation

**Quote fragments, not phrases.** `melody._phrase_tokens` gets the rule
practice already follows (`DECISIONS.md`, "Originality is enforced"): a
learned phrase longer than the originality checker's maximum shared run
(`DEFAULT_MAX_RUN`, six notes) is quoted as a fragment transposed into the
current octave, never whole; a phrase within the limit may be quoted whole. This ends the quote-then-reject
loop in `generate_tune` for the common case; the loop stays as the safety
net. Authenticity credit in the evaluator comes from characteristic pairs
and short exact matches, so fragment quoting keeps authenticity while
satisfying originality.

**An idiom model from the phrase bank** (section 37). New
`agent/idiom.py`: `RaagaIdiom.from_phrases(phrases)` computes, weighted by
phrase confidence: degree-transition preferences (which scale step follows
which, ascending and descending), leap frequency, typical phrase length,
cadence endings and their shares, and contour shares (rise / fall / arch).
`learned_raaga()` attaches it to the learned `Raaga` view; the library view
has none. `_phrase_tokens` consults it, when present, in exactly the places
it currently flips a coin: the direction draw, the step-versus-leap draw,
and the cadence choice. With no idiom attached, the draws are unchanged and
the golden melodies in `tests/golden/` stay byte-identical.

**Guided regeneration in `generate_tune`.** Today the job regenerates only
for originality. It will evaluate the candidate with the agent's evaluator,
and if the overall score is under the composing threshold, regenerate with
`Guidance` built from the candidate's findings plus the raaga's stored
lessons, three tries, keeping the best. The action reports its phases through
the section 6.1 contract ("Writing phrases", "Listening back", "Rewriting:
<reason>"), so a rewrite is visible rather than silent.

**Provenance for "why did you choose this phrase?"** (section 16). Each
quoted fragment is recorded on the `MelodyVersion` as `(note range, phrase
id, source)`. `MusicAgent.explain` answers the question from that record and
the lesson history, and LEARN History's "composition learning history" lists
what each tune borrowed and what it was told to avoid.

**Creator feedback closes the loop** (sections 26, 27, 39). `give_feedback`
already lowers or raises phrase confidence. It will also write a lesson with
the finding kinds the evaluator raises for the same tune, so "too
mechanical" on a tune whose critique says `repetitive` and `no_gamaka`
becomes guidance (`vary_more`, `add_gamaka`) for the next tune in that
raaga. The next tune is different for a reason the agent can state.

## Queue

Each item is one branch and one PR; each is verified on its own before the
next starts. Files are named so two items can run side by side only when
they do not overlap.

| # | Task | Touches | Verification |
|---|---|---|---|
| 1 | Structured findings and the lessons table. Evaluator raises `Finding`s next to its strings; repository gains `lessons` with the section 38 fields; `_practice_step`, `critique_tune`, `give_feedback` write lessons; LEARN shows them (Practice remediation list, Knowledge gaps, History findings). No change to any generated note. | `agent/evaluator.py`, `agent/knowledge.py`, `agent/music_agent.py`, `app.py`, `ui/learn_workspace.py`, new tests | Unit tests per finding kind; a restart test (section 27: learn, restart, retrieve) proving lessons survive; full suite unchanged in count of generated-note assertions |
| 2 | Guidance in practice. `agent/guidance.py`; `_motif` and `_notes_from_tokens` obey it; recurrence counting. | new `agent/guidance.py`, `agent/practice.py`, `agent/music_agent.py`, tests | REG test: on a fixed seed where attempt 0 fails with finding X, attempt 1 does not exhibit X and scores higher; a property test over 40 seeds that guided retries never score lower on the guided dimension; a test that an empty `Guidance` reproduces today's tokens exactly |
| 3 | Composer: fragment quoting and the idiom model. `RaagaIdiom` from the phrase bank; attached by `learned_raaga`; `_phrase_tokens` uses it only when present. | new `agent/idiom.py`, `agent/learned.py`, `music/melody.py`, `raaga/library.py` (optional `idiom` field), tests | Golden melodies byte-identical (library raaga, no idiom); measured before/after table of the twelve dimensions on Keeravani tunes for seeds 1..20 with a knowledge base filled by the reference provider; the originality regeneration loop fires fewer times (counted in the log) |
| 4 | Guided regeneration and provenance in `generate_tune`; `explain` answers "why this phrase"; feedback writes lessons. | `app.py`, `agent/music_agent.py`, `core/models.py` (provenance on `MelodyVersion`; old projects still load), `ui` (History), tests | TEST-style acceptance: generate, critique, say "too mechanical", generate again: the second tune's `repetitive` finding is gone and the agent's explanation names the lesson; project save/reopen keeps provenance |
| 5 | Docs and calibration. `DECISIONS.md` entry; the "known weakness" paragraph replaced by the measured numbers from item 3; `PLAN_v0.3.md` cross-reference; live verification against a scratch config directory. | `docs/` | Live: fresh scratch home, "Learn Keeravani" for 60 steps, generate, critique, feedback, regenerate; screenshots of the Practice remediation list and the History findings |

Order matters: 1 before 2 and 4 (both read lessons); 3 is independent of 1
and 2 and can run alongside them; 4 needs 1, 2 and 3.

## Provisional decisions

* **Guidance constrains; it never supplies the answer.** A finding says
  "the line ended on Ri, and Ri is not a resting note"; guidance says "end
  on a nyasa"; the generator still chooses which. The evaluator is not
  told what guidance was applied. Otherwise a pass would measure the
  guidance, not the agent.
* **Lessons are per raaga with a unit bonus, not per unit only.** A mistake
  in `b13.short_phrase` (leaping too much) is the same mistake in
  `b14.chains` and in a tune. Section 38's "related knowledge" field is what
  links them.
* **Determinism stays.** Guidance and the idiom model are pure functions of
  the knowledge base, and seeds stay `practice_seed(unit, attempt)`. The same
  knowledge base and the same attempt produce the same line in every
  process. REG-095 and REG-100 remain the guard.
* **Nothing changes for a raaga the agent has not studied.** The idiom model
  exists only on the learned view, so a library raaga composes exactly as it
  does today and the golden files are the proof. This is section 0's "map
  onto what exists" applied to the composer.
* **Fragment quoting follows the practice rule already recorded in
  `DECISIONS.md`** rather than a new number: whole quotation only within the
  originality checker's run limit. One rule for practice and composition.
* **The composing threshold for guided regeneration is the evaluator's
  existing overall score, not a new metric.** It is bounded to three
  rewrites, the best candidate is kept, and a rewrite is reported as a phase,
  so a long job never looks like a hang (section 6.1).
* **Not built here:** an LLM critic (the local evaluator is the teacher; a
  provider-backed critique is a later provider-routing item), learning from
  real recordings (separate: run one real Keeravani recording through
  Training and read the confidences), and Stage C cross-raaga work (needs a
  second mastered raaga).

## Hard blockers

None. Every step is local, deterministic and covered by the existing suite
plus the named tests.

## Evidence this will be judged by

Before any code, the baseline is recorded so improvement is measured, not
claimed:

| Measurement | How |
|---|---|
| Twelve evaluator dimensions on Keeravani tunes, seeds 1..20, knowledge base filled by the reference provider | script under `tools/`, table into `DECISIONS.md` |
| Share of practice retries that repeat a finding of the same kind as the previous attempt, over the 35 units and 5 seeds | same script; the number section 38 is about |
| Originality regenerations per tune | log count |
| Steps to complete the Keeravani curriculum, and the pass rate of first attempts | the REG-095 harness |

The plan is done when the first two numbers move in the right direction with
the suite green, and a creator can ask the agent why it played a phrase and
get an answer that cites what it learned and what it was told.
