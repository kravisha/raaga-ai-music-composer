# Agent Factory: the Universal Learning Framework, piloted on the Raga agent

Written 2026-09-04 against the seven-part "Agent Factory - Universal Learning
Framework v0.1" (kept verbatim under `docs/spec/agent_factory/`) and the
canonical v0.3 specification. Same shape as the other plans: what is asked,
what exists, the design, the queue, the decisions, the evidence.

## What the framework asks

| Document | Requirement | State today |
|---|---|---|
| 01 P1, P7 | Learning cannot advance without evidence; promotion by demonstrated capability | The curriculum advances on a practice score. No separate test record, no split between what was practised and what proves it |
| 01 section 3, 4 | A Lesson object with objective, examples, counterexamples, practice, tests, common errors, remediation; a ten-step cycle | Curriculum units carry goal, exercises, thresholds. No reiteration, no explanation step |
| 01 section 5 | Mastery levels L0 to L9 per concept, partial mastery allowed | `UnitProgress`: not started, in progress, passed, failed |
| 02 | Student, Trainer and temporary Judge roles; Trainer also learns | `MusicAgent` is student and examiner in one; the evaluator is the only critic |
| 03 | Test ladder T0 to T10; adaptive difficulty; retire defeated tests; four data splits; promotion gate | Fixed exercises per unit, seeds per attempt (REG-100). No ladder, no splits |
| 04 | Knowledge classes; shared versus agent memory versus training history; reiteration protocol R1 to R8; memory promotion; deprecation fields | One knowledge base per agent; lessons table (item 1); facts have confidence and provenance but no class, version or supersession |
| 05 | Factory input spec; bootstrap from shared knowledge; maturity S0 to S7; release gate; field learning; factory feedback | None |
| 06 | Ten acceptance tests; Raga agent as first Student with a Trainer; do not generalise prematurely | The Raga agent, curriculum, practice, evaluator and lessons exist to be adapted |

The framework's own handoff sets the order: integrate into the Raga workflow first, validate the loop, then extract. The design below keeps the reusable part in one package from the start, because the acceptance tests demand a second agent that inherits a lesson (TEST 6) and a domain-independent architecture (implementation principle), but every judgment about music lives in the adapters.

## Design

### `raagacomposer/factory/`: the domain-independent core

**models.py** holds the data objects the handoff lists, as dataclasses with enums for the ladders: `MasteryLevel` L0 to L9, `TestLevel` T0 to T10, `Maturity` S0 to S7, `KnowledgeClass` (hard, heuristic, experience, procedure, test, dispute_lesson), `Split` (training, validation, hidden, real_world), `DisputeStatus`. Objects: `AgentSpec` (factory input), `AgentProfile`, `Lesson`, `Reiteration`, `TestSpec`, `Performance` (what a student hands back: output, claim, confidence, evidence), `TestResult`, `Dispute`, `Ruling`, `ReusableLesson`, `MasteryRecord`, `Promotion`, `Remediation`, `GateReport`.

**protocols.py** defines the three roles as protocols, so an adapter is anything with these methods:

- `Student.profile`, `acquire(lesson)`, `reiterate(lesson) -> Reiteration`, `perform(test) -> Performance`, `apply_correction(text, lesson)`.
- `Trainer.next_lesson(profile, history)`, `build_tests(lesson, profile, history) -> List[TestSpec]`, `grade(test, performance) -> TestResult` (carries the trainer's claim and confidence), `check_reiteration(lesson, reiteration) -> ReiterationCheck`, `remediate(profile, lesson, failures) -> Remediation`, `learn_from(result)`.
- `Rule`: `name`, `knowledge_class`, `applies(dispute)`, `decide(dispute, knowledge) -> Optional[Ruling]`. Rules are how hard knowledge reaches the Judge.

**store.py**: `FactoryStore`, one sqlite file (`settings.factory_db`, default `config_dir()/factory.db`), schema-versioned like the knowledge repository, every public method under an `RLock`. Tables: profiles, lessons, reiterations, tests, results, disputes, rulings, reusable_lessons, mastery, promotions, metrics. This is the SHARED KNOWLEDGE BASE and the TRAINING HISTORY of document 04. AGENT MEMORY stays where it is: the agent's own `knowledge.db`.

**mastery.py**: evidence rules. Exposure gives L1. A reiteration the trainer accepts gives L2 (restate) and L3 (explain). Passing T3 gives L4, T4 on a validation-split test gives L5, T6 or T7 gives L6, T8 or T9 gives L7, a test the agent authored that another agent then used gives L8, real-world evidence gives L9. Three failures at the current level's test drop one level. Without a passed application test, mastery is capped at L3 (TEST 7, no false mastery).

**trainer.py**: `AdaptiveTrainer`, the generic half of any trainer. Chooses the next test level from mastery; raises difficulty when the last three results at the level passed with confidence within 0.2 of the score; remediates when the same failure mode appears twice running, and a remediation must change something (guided practice, a different exercise family, a lower level), never the same test again; retires a test the student has beaten three times to the regression split and asks the domain trainer for a harder or more novel variant; assigns splits so that hidden tests are never used for practice. A domain trainer supplies the tests; this class decides which to give.

**judge.py**: `convene(dispute, knowledge, rules, escalate=None) -> Ruling`. Builds a temporary judge holding both evidence sets, the relevant shared knowledge and the applicable rules; tries hard rules first, then the escalation callable if one is configured (a provider-backed model, later), and otherwise returns `UNRESOLVED` with the question and what evidence would settle it. The judge object is local to the call. The cycle persists the dispute, the ruling and any reusable lesson; nothing persists the judge (TEST 5).

**cycle.py**: `LearningCycle.run(lesson)` executes document 01's ten steps: acquire, reiterate, explain and connect (checked by the trainer), apply, test, dispute if claims conflict with both plausible (both confidences at or above 0.5, gap at most 0.25, no hard rule settles it), resolve, persist, advance or remediate. Returns a `CycleOutcome` with the mastery change and what happens next.

**gates.py**: `promotion_gate` (document 03 section 7: mastery threshold, no critical unresolved failure, success on unseen tests, stable repeats, calibrated confidence) and `release_gate` (document 05 section 5: hidden tests, no critical unresolved dispute, rollback, bounded permissions, monitoring, escalation, knowledge version). The deployer states rollback, permissions, monitoring and escalation in the `AgentSpec`; the gate checks they are stated and that the evidence tables support the rest.

**factory.py**: `AgentFactory.create(spec, curriculum, initial_tests)` bootstraps a profile with the validated reusable lessons in scope preloaded; `train(student, trainer, max_cycles)`; `assess(profile)` walks the maturity pipeline S0 to S7 from evidence; `field_lesson(profile, event)` turns real-world experience into a candidate lesson that must be validated before it is shared (document 05 section 6); `metrics()` records what the factory learns: which tests predicted later success, which mistakes recur across agents, which lessons were worth preloading.

A toy domain under `tests/unit/factory/` (a scripted student learning a small rule set) proves the core without music and carries acceptance tests 5, 6, 7, 9 and 10 in domain-free form.

### The Raga pilot: adapters in `raagacomposer/agent/`

**student.py**: `RagaStudent` wraps a `MusicAgent`. Reiteration comes from the knowledge base, not a prompt: restate is the stored fact in its own words; explain says what the fact is for (jeeva carry the raaga, nyasa are where phrases rest, the arohanam is the permitted way up); connect names the passed prerequisites and related facts; example is the best-trusted phrase; counterexample is a corrupted phrase with the reason it is wrong (the practice engine already builds these); apply is a practice run; self-check lists the weakest evaluator dimensions and the open lessons; retest is the curriculum's revisit. `perform` maps the ladder onto the practice engine: T0 and T1 to listen.identify and recall.fact, T2 to the explanation check, T3 to generate.pattern with guidance, T4 without, T5 to variations, T6 to classify.valid, T7 to a new correct-the-phrase exercise, T8 to the same pattern in a second raaga or register, T9 to neighbour drift, T10 to generate.section and, in item 4, to Generate Tune. The student's claim is its own judgement: for classification its answer, for generation its self-evaluation from the learned view; its confidence is the evaluator's confidence or the knowledge confidence.

**trainer.py**: `RagaTrainer` builds framework lessons from curriculum units plus the knowledge base (explanation from the library and facts, examples from phrases, counterexamples by corruption, common errors from the lessons table, remediation from guidance kinds). It generates tests by level with novelty through seeds the student never practised on (`practice_seed` gains a salt; hidden tests use a disjoint salt), tighter tolerances, longer lines, originality checks on, a second raaga for T8. It grades with the evaluator and the library's hard rules; its claim and confidence are the evaluation's. It remediates through `Guidance` (guided T3), a different exercise family, or a level down. It learns by retiring beaten tests and recording every result's difficulty, novelty and failure mode.

**rules.py**: hard rules for the Judge from the library: allowed swaras, ascent and descent order, forbidden swaras, resting notes, plus the originality index. A learned fact that contradicts the library is heuristic knowledge and loses to a hard rule; a dispute a hard rule cannot settle (a mood claim, a phrase's authenticity) goes to escalation or stays unresolved.

**Where disputes actually arise.** The student classifies with its learned view; the trainer with the library and evaluator. When a heard fact disagrees with the library (the knowledge base already flags disputed facts), the two claims diverge with plausible evidence on both sides. That is the case the Judge exists for, and it happens in the real data.

**MusicAgent integration.** `MusicAgent.factory` holds the store and the agent's profile ("raga-agent", domain "carnatic-music"). `train_step()` runs one learning cycle on the next unit; `learn_step()` keeps its present behaviour so every existing test and the running app are unchanged. LEARN's Practice area gets the ladder view: per-concept mastery, the last tests with level and novelty, remediation in force, open disputes and rulings, maturity and gate status.

### Item 4, integrated

Generate Tune is the T10 test. `generate_tune` evaluates each candidate, regenerates with guidance from its findings and the raaga's lessons when under the composing threshold (three tries, best kept, each rewrite a reported phase), records provenance for every quoted fragment, and files a `TestResult` at T10 against the profile. Creator feedback is field evidence: negative feedback opens a dispute between the student's self-evaluation and the creator's verdict, resolved in the creator's favour with a reusable lesson and an L9 signal; positive feedback validates. `explain` answers "why this phrase" from provenance and lessons. History under LEARN lists composition learning history.

## Queue

| # | Branch | Task | Touches | Proves |
|---|---|---|---|---|
| F1 | `factory/1-core` | The core package with models, protocols, store, mastery, adaptive trainer, judge, cycle, gates, factory; toy domain | new `raagacomposer/factory/`, `core/settings.py` (factory_db), `tests/unit/factory/` | Acceptance 5, 6, 7, 9, 10 domain-free; 2, 3, 4, 8 on the toy domain |
| F2 | `factory/2-raga-pilot` | RagaStudent, RagaTrainer, rules, `practice_seed` salt, correct-the-phrase exercise, `MusicAgent.train_step`, LEARN ladder view | `agent/student.py`, `agent/trainer.py`, `agent/rules.py`, `agent/practice.py`, `agent/music_agent.py`, `ui/learn_workspace.py`, tests | Acceptance 1, 2, 3, 4, 8 on the Raga agent; REG-095 and every existing test unchanged |
| F3 | `factory/3-compose-and-feedback` | Item 4: guided regeneration, provenance, explain, feedback as field evidence and T10 results | `app.py`, `agent/music_agent.py`, `core/models.py`, History view, tests | The item 4 acceptance test; save and reopen keeps provenance |
| F4 | `factory/4-docs` | DECISIONS entry, PLAN cross-references, live verification | `docs/` | Screenshots of the ladder view and a dispute resolved |

F1 and F2 run side by side: F2 codes against `models.py` and `protocols.py`, which are written first and frozen. F3 needs both.

## Provisional decisions

* **The core is a package now, not extracted later**, because TEST 6 needs a second agent and the toy domain is the cheapest proof of independence. Nothing musical is imported by `factory/`.
* **The Judge is a function that builds a throwaway object**, so ephemerality (TEST 5) is structural: there is no class instance to persist.
* **Hard knowledge is the library; heard knowledge is heuristic or experience.** Facts get a knowledge class; only hard rules decide disputes; a heuristic never overrides a hard rule.
* **No model call in the loop by default.** Reiteration, tests, grading and rulings are built from stored knowledge and the existing engines, so the suite runs offline and deterministically. The escalation hook exists for a provider-backed judge when the creator configures one.
* **Existing behaviour is untouched.** `learn_step` and the curriculum keep working; the cycle wraps them. The 1058 tests stay green.
* **Deferred:** a second musical agent (lyrics) bootstrapped from the raga agent's reusable lessons is the framework's next pilot, not this work; TEST 6 is proven with the toy domain and with a second raga profile.

## Evidence

Every acceptance test in document 06 becomes a named pytest test (`test_af01_reiteration` to `test_af10_factory_release`). The measurement script from the learning-loop plan is re-run after F3. Live verification in F4 runs a training session in the app against a scratch configuration directory.

## Status, 2026-09-04

| Increment | Pull request | Outcome |
|---|---|---|
| F1, F2 | #9 (one branch, three commits) | Core and Raga pilot; a probe of real training cycles found four seam defects and one latent learned-view bug, fixed before the PR |
| F3 | #10 | Item 4 folded in: the tune as the T10 test, the creator as external evidence |
| F4 | this branch | Decisions recorded, plans cross-referenced, the Stage 1 knowledge pack kept and checked (`docs/PLAN_stage1_knowledge.md`), live pass with screenshots |

Where each acceptance test lives:

| Document 06 test | Domain-free (toy) | Raga agent |
|---|---|---|
| 1 reiteration | | `test_af01_reiteration` |
| 2 adaptive testing | `test_af02_adaptive_testing` | `test_af02_adaptive_testing` |
| 3 remediation | `test_af03_remediation` | `test_af03_remediation` |
| 4 judge | `test_af04_judge` | `test_af04_judge`, `test_af04b_a_wrong_belief_is_corrected_through_a_training_cycle` |
| 5 judge ephemerality | `test_af05_judge_ephemerality` | |
| 6 knowledge reuse | `test_af06_knowledge_reuse` | |
| 7 no false mastery | `test_af07_no_false_mastery` | |
| 8 test evolution | `test_af08_test_evolution` | `test_af08_test_evolution` |
| 9 hard vs heuristic | `test_af09_hard_vs_heuristic` | `tests/unit/test_rules.py` |
| 10 factory release | `test_af10_factory_release` | |
| field evidence | | `test_af_field_evidence_can_reach_l9` |

Deferred, as the plan said: a second musical agent bootstrapped from the raga agent's reusable lessons, and a provider-backed Judge behind the escalation hook.
