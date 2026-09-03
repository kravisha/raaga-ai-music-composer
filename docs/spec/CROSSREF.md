# Cross-Reference: Old Specification Sections to CANONICAL_SPEC_v0.3

This file maps references to four older specification documents to their corresponding sections in the consolidated canonical specification (v0.3, dated 2026-09-03). The old documents are still cited throughout the codebase; this reference lets readers find the current home of each topic.

Code comments retain their original section numbers on purpose: rewriting every reference would introduce churn without changing behavior. Documentation comments and prose should cite v0.3 sections. The tables below show where each old topic now lives and which parts of the codebase care about it.

## Table 1: Old References to v0.3 Mappings

| Old reference | Old document (best guess) | Where it is cited (file:line, comma-separated) | Topic | v0.3 section(s) |
|---|---|---|---|---|
| section 2 | Master Spec v0.2 | docs/DECISIONS.md:3 | Reversible engineering decisions taken without interruption | 65 |
| section 4 | Master Spec v0.2 | raagacomposer/lyrics/fitting.py:1 | Lyric-to-melody fitting engine | 9 |
| section 4 step 2 | Master Spec v0.2 | raagacomposer/raaga/selection.py:1 | Raaga selection engine | 7 |
| section 4 step 4 | Master Spec v0.2 | raagacomposer/lyrics/fitting.py:1 | Lyric fitting to tune | 9 |
| section 5.4 | Master Spec v0.2 | raagacomposer/speech/context.py:1 | Conversational context manager | 13 |
| section 6 | Master Spec v0.2 | raagacomposer/speech/timeline_parser.py:1 | Natural-language timeline parser | 13.3 |
| section 7 | Master Spec v0.2 | raagacomposer/music/instruments.py:1 | Instrument catalog | 11 |
| section 7.2 | Master Spec v0.2 | raagacomposer/music/instruments.py:347 | Instrument ranking against feel words | 11 |
| section 9 | Master Spec v0.2 | raagacomposer/music/structure.py:1 | Song structure planning with templates and cycle scaling | 12 |
| section 10 | Master Spec v0.2 | raagacomposer/raaga/library.py:1 | Raaga knowledge store (data/raagas.json) | 37 |
| section 11 | Master Spec v0.2 | raagacomposer/providers/base.py:1, raagacomposer/providers/registry.py:1 | Provider abstraction layer and registry | 41 |
| section 12.2 | Master Spec v0.2 | raagacomposer/app.py:1 | Application controller as single point of project state change | 1, 64 |
| section 12.33 | Master Spec v0.2 | raagacomposer/core/settings.py:1 | Settings and credential management | 42, 55 |
| section 14 | Master Spec v0.2 | raagacomposer/ui/main_window.py:1 | Main desktop window | 4 |
| section 14A | Master Spec v0.2 | raagacomposer/ui/panels/project_panel.py:1 | Project header panel | 4.1 |
| section 14B | Master Spec v0.2 | raagacomposer/ui/panels/brief_panel.py:1 | Creative brief panel | 5, 6 |
| section 14C | Master Spec v0.2 | raagacomposer/ui/panels/raaga_panel.py:1 | Raaga panel | 7 |
| section 14E | Master Spec v0.2 | raagacomposer/ui/panels/lyrics_panel.py:1 | Lyrics panel | 9 |
| section 19 | Master Spec v0.2 | raagacomposer/audio/export.py:1 | Export engine (WAV, MP3, MIDI, MusicXML, stems) | 57 |
| section 20 rule 1 | Training spec | docs/DECISIONS.md:329 | User approval is mandatory; searching is not approving | 20 |
| section 21 | Master Spec v0.2 | README.md:408 | Acceptance scenario: create project through reopen and confirm state | 63 |
| section 3.1, 3.2 | Training spec | raagacomposer/training/search.py:1 | Finding and ranking material worth learning from | 18 |
| section 3.3 | Training spec | raagacomposer/training/queue.py:4, raagacomposer/training/models.py:53 | Learning queue: statuses, one source at a time, persistent | 19 |
| section 5 | Training spec | raagacomposer/training/search.py:1, raagacomposer/training/models.py:158 | Training source search results normalized | 18 |
| section 6 | Training spec | raagacomposer/training/objectives.py:1, raagacomposer/training/models.py:199 | Learning objectives per source | 21 |
| section 8 | Training spec | raagacomposer/training/report.py:51, raagacomposer/training/models.py:303 | Learning report structure with all required sections | 23 |
| section 9 | Training spec | raagacomposer/training/models.py:220 | Learned items with provenance, source, run, confidence | 30 |
| section 15, 16 | Training spec | raagacomposer/training/store.py:1 | Durable storage across close/restart; every completed source has a report | 46, 47 |
| section 3.1, 9, 15, 16 | Learning spec | raagacomposer/agent/music_agent.py:1 | Music agent: student behind instrument, orchestrator, learning loop | 16 |
| section 3.2, 4, 5 | Learning spec | raagacomposer/agent/curriculum.py:1 | Curriculum engine with executable data and stages | 17, 17.1-17.4 |
| section 7 | Learning spec | raagacomposer/agent/analysis.py:1 | Audio ingestion and music analysis pipeline | 22 |
| section 8 | Learning spec | raagacomposer/agent/knowledge.py:1 | Permanent knowledge repository that survives restart | 28 |
| section 12 | Learning spec | raagacomposer/agent/originality.py:1 | Originality safeguards: phrase fingerprinting, rejection of near-duplicates | 51 |
| section 13 | Learning spec | raagacomposer/agent/practice.py:1 | Practice engine: agent plays, listens, practices lessons | 24 |
| section 14 | Learning spec | raagacomposer/agent/evaluator.py:1 | Teacher/evaluator: twelve separate dimension scores, no collapse | 26 |
| section 17 | Learning spec | raagacomposer/ui/panels/agent_panel.py:1 | Music agent/learning panel (now LEARN workspace) | 4.2 |
| section 4 to 13 | KB architecture spec | raagacomposer/kb/models.py:1 | Knowledge item, claim, entity, evidence, relationship models | 30-32 |
| section 8, 15 | KB architecture spec | raagacomposer/kb/normalize.py:1 | Duplicate-control key and identity for claims | 29, 35 |
| section 9 | KB architecture spec | raagacomposer/kb/models.py:167 | Evidence: source, timestamp, segment, feature reference, strength | 33 |
| section 10, 41 | KB architecture spec | raagacomposer/kb/confidence.py:1 | Confidence with visible reasoning; eight factors tracked not collapsed | 34 |
| section 18 | KB architecture spec | raagacomposer/kb/retrieval.py:1 | Getting knowledge back out: hybrid retrieval, ranked for usefulness | 36 |
| section 19 | KB architecture spec | raagacomposer/kb/context.py:1 | Assembling what a task actually needs from the KB | 36 |
| section 26 | KB architecture spec | docs/DECISIONS.md:371, raagacomposer/kb/migrate.py:1 | Learning Report (what happened in one run) vs Knowledge Base (what accumulates) | 23, 28 |
| section 30, 39, 40 | KB architecture spec | raagacomposer/kb/librarian.py:1 | Librarian: organize, detect duplicates, maintain aliases, detect gaps | 50, 40 |
| section 36 | KB architecture spec | docs/DECISIONS.md:389 | Corrupted KB is kept with timestamped copy; never silently replaced | 36, 55 |
| section 39 | KB architecture spec | raagacomposer/kb/librarian.py:210 | Linking and organizing knowledge graph | 50 |
| section 40 | KB architecture spec | raagacomposer/kb/librarian.py:304 | Knowledge gap detection and health reporting | 40 |
| section 47 | KB architecture spec | docs/DECISIONS.md:347, raagacomposer/app.py:309, raagacomposer/kb/migrate.py:1 | Inspect existing repo first; migrate rather than recreate | 0, 47 |
| section 20 phase D | Master Spec / Training spec | docs/DECISIONS.md:78 | Timeline range specification: "from the second minute to the third minute" = 01:00-03:00 | 13.3 |
| section 20 rule 10 | Training spec | raagacomposer/training/access.py:8 | System does not bypass paywall, DRM, or access controls | 20 |

## Table 2: v0.3 Sections and Their Old Mappings

| v0.3 section | Title | Old references that map here | Implemented in (module paths) |
|---|---|---|---|
| 0 | Status, Authority, Purpose | KB spec section 47 | raagacomposer/kb/migrate.py |
| 1 | Governing Development Principle | Master Spec v0.2 section 12.2 | raagacomposer/app.py, raagacomposer/core/models.py |
| 2 | Core Product Vision | (foundational; no old mapping) | |
| 3 | Platform Requirements | (foundational; covered by multiple old sections) | |
| 3.1 | Desktop Application | (foundational) | raagacomposer/ui/main_window.py |
| 3.2 | Python | (foundational) | raagacomposer/app.py, entire package |
| 3.3 | Packaging | (foundational) | packaging/ |
| 4 | Top-Level UI Architecture | Master Spec v0.2 section 14 | raagacomposer/ui/main_window.py |
| 4.1 | Main Workspace | Master Spec v0.2 section 14A | raagacomposer/ui/main_window.py, raagacomposer/ui/panels/project_panel.py |
| 4.2 | Learn Workspace | Learning spec section 17 | raagacomposer/ui/panels/agent_panel.py |
| 4.3 | Menu Bar | (foundational; no explicit old reference) | raagacomposer/ui/main_window.py |
| 5 | Main Workspace - Creative Brief | Master Spec v0.2 section 14B | raagacomposer/ui/panels/brief_panel.py |
| 6 | Apply Brief | Master Spec v0.2 section 14B (implied) | raagacomposer/ui/panels/raaga_panel.py, raagacomposer/app.py |
| 6.1 | Required Action Status Contract | (v0.3 new) | raagacomposer/core/actions.py |
| 7 | Raga Selection | Master Spec v0.2 section 14C | raagacomposer/ui/panels/raaga_panel.py, raagacomposer/raaga/selection.py |
| 8 | Tune-First Workflow | Master Spec v0.2 (implicit in composition flow) | raagacomposer/music/melody.py, raagacomposer/ui/panels/tune_panel.py |
| 9 | Lyrics | Master Spec v0.2 sections 4, 14E | raagacomposer/lyrics/fitting.py, raagacomposer/lyrics/generator.py, raagacomposer/ui/panels/lyrics_panel.py |
| 10 | Voice / Singer | Master Spec v0.2 (composition workflow) | raagacomposer/voice/profiles.py, raagacomposer/voice/renderer.py, raagacomposer/ui/panels/voice_panel.py |
| 11 | Arrangement / Instrumentation | Master Spec v0.2 sections 7, 7.2 | raagacomposer/music/instruments.py, raagacomposer/music/arrangement.py, raagacomposer/ui/panels/arrangement_panel.py |
| 12 | Track / Timeline Model | Master Spec v0.2 section 9 | raagacomposer/music/structure.py, raagacomposer/ui/timeline.py |
| 13 | Continuous Conversational Voice Control | Master Spec v0.2 section 5.4 | raagacomposer/speech/context.py |
| 13.1 | Interruption / Barge-in | Master Spec v0.2 (implied in voice control) | raagacomposer/core/jobs.py, raagacomposer/app.py |
| 13.2 | Contextual References | Master Spec v0.2 section 5.4 | raagacomposer/speech/context.py |
| 13.3 | Natural Language Playback | Master Spec v0.2 sections 6, 20 phase D | raagacomposer/speech/timeline_parser.py, raagacomposer/audio/playback.py |
| 14 | Self-Learning Music Agent | Learning spec section 3.1 | raagacomposer/agent/music_agent.py |
| 15 | Core Learning Components | Learning spec (foundational learning architecture) | raagacomposer/agent/ |
| 16 | Music Agent / Orchestrator | Learning spec sections 3.1, 9, 15, 16 | raagacomposer/agent/music_agent.py |
| 17 | Curriculum Engine | Learning spec sections 3.2, 4, 5 | raagacomposer/agent/curriculum.py |
| 17.1 | Stage A - Universal Foundations | Learning spec section 3.2 | raagacomposer/agent/curriculum.py |
| 17.2 | Stage B - One Raga at a Time | Learning spec section 3.2 | raagacomposer/agent/curriculum.py |
| 17.3 | Stage C - Cross-Raga Comparison | Learning spec section 3.2 | raagacomposer/agent/curriculum.py |
| 17.4 | Curriculum Unit Data | Learning spec section 5 | raagacomposer/agent/curriculum.py |
| 18 | Training Source Search | Training spec sections 3.1, 3.2, 5 | raagacomposer/training/search.py |
| 19 | Learning Queue | Training spec section 3.3 | raagacomposer/training/queue.py |
| 20 | Source Access Policy | Training spec sections 20 rule 1, rule 10 | raagacomposer/training/access.py |
| 21 | Learning Objectives | Training spec section 6 | raagacomposer/training/objectives.py |
| 22 | Content Ingestion / Music Analysis Pipeline | Learning spec section 7 | raagacomposer/agent/analysis.py, raagacomposer/training/pipeline.py |
| 23 | Learning Report | Training spec section 8, KB spec section 26 | raagacomposer/training/report.py |
| 24 | Practice Engine | Learning spec section 13 | raagacomposer/agent/practice.py |
| 25 | Training Agent / Quiz Behavior | Learning spec (teacher/evaluation component) | raagacomposer/agent/evaluator.py |
| 26 | Evaluator / Teacher / Critic | Learning spec section 14 | raagacomposer/agent/evaluator.py |
| 27 | Complete Learning Loop | Learning spec (foundational loop architecture) | raagacomposer/agent/music_agent.py |
| 28 | Persistent Knowledge Base | Learning spec section 8, KB spec section 26 | raagacomposer/agent/knowledge.py, raagacomposer/kb/store.py |
| 29 | Knowledge is a Network | KB architecture spec section 3 (core idea) | raagacomposer/kb/service.py, raagacomposer/kb/models.py |
| 30 | Knowledge Item / Claim Model | KB architecture spec sections 4-13, Training spec section 9 | raagacomposer/kb/models.py |
| 31 | Knowledge Types | KB architecture spec (type taxonomy) | raagacomposer/kb/schema.py |
| 32 | Relationships | KB architecture spec (implied in network model) | raagacomposer/kb/models.py, raagacomposer/kb/service.py |
| 33 | Source Provenance / Evidence | KB architecture spec section 9 | raagacomposer/kb/models.py |
| 34 | Confidence / Contradiction / Versioning | KB architecture spec sections 10, 41 | raagacomposer/kb/confidence.py, raagacomposer/kb/service.py |
| 35 | Duplicate Control / Compaction | KB architecture spec sections 8, 15 | raagacomposer/kb/normalize.py, raagacomposer/kb/service.py |
| 36 | Memory / Performance / Speed | KB architecture spec sections 18, 19, 36 | raagacomposer/kb/retrieval.py, raagacomposer/kb/context.py |
| 37 | Raga Knowledge Model | Master Spec v0.2 section 10 | raagacomposer/raaga/library.py |
| 38 | Tala / Procedural / Example / Failure Knowledge | KB architecture spec (knowledge types) | raagacomposer/kb/schema.py |
| 39 | User Correction | KB architecture spec section 39 | raagacomposer/kb/service.py |
| 40 | Knowledge Gap Detection | KB architecture spec section 40 | raagacomposer/kb/librarian.py |
| 41 | Provider Manager | Master Spec v0.2 section 11 | raagacomposer/providers/base.py, raagacomposer/providers/registry.py |
| 41.1 | Provider Abstraction | Master Spec v0.2 section 11 | raagacomposer/providers/base.py |
| 41.2 | Routing | Master Spec v0.2 section 11 | raagacomposer/providers/router.py, raagacomposer/providers/tasks.py |
| 41.3 | Fallback | Master Spec v0.2 section 11 (implied) | raagacomposer/providers/router.py |
| 41.4 | Offline Mode | Master Spec v0.2 section 11 (implied) | raagacomposer/providers/router.py |
| 42 | Anthropic / Claude API Key | Master Spec v0.2 section 12.33 | raagacomposer/core/settings.py |
| 43 | Local SLM Support | Master Spec v0.2 section 11 | raagacomposer/providers/registry.py, raagacomposer/providers/ollama_llm.py |
| 44 | Deterministic Logic vs Model Logic | (architectural principle; foundational) | raagacomposer/providers/router.py |
| 45 | Background Learning | Learning spec section 16 | raagacomposer/agent/music_agent.py |
| 46 | Persistence / Checkpointing | Training spec sections 15, 16, Learning spec section 20 | raagacomposer/core/persistence.py, raagacomposer/training/store.py |
| 47 | Database Direction | KB architecture spec section 47, Master Spec v0.2 section 12.2 | raagacomposer/kb/store.py, raagacomposer/core/persistence.py |
| 48 | Recommended Core Tables / Collections | (implementation detail derived from KB spec) | raagacomposer/kb/schema.py |
| 49 | Knowledge Base Service | KB architecture spec (implied in KB design) | raagacomposer/kb/service.py |
| 50 | Librarian / Knowledge Manager | KB architecture spec sections 30, 39, 40 | raagacomposer/kb/librarian.py |
| 51 | Originality | Learning spec section 12 | raagacomposer/agent/originality.py |
| 52 | Project Data Model / Versioning | Master Spec v0.2 section 12.2 | raagacomposer/core/models.py, raagacomposer/core/versioning.py |
| 53 | Background Jobs / Cancellation | Master Spec v0.2 section 12.2 (implied) | raagacomposer/core/jobs.py |
| 54 | Error Handling / Diagnostics | (foundational; cross-cutting) | raagacomposer/core/logging_setup.py |
| 55 | Security | Master Spec v0.2 section 12.33 | raagacomposer/core/settings.py, raagacomposer/core/logging_setup.py |
| 56 | Explainability | Learning spec section 14 (evaluation includes explanation) | raagacomposer/agent/evaluator.py, raagacomposer/kb/context.py |
| 57 | Output / Export | Master Spec v0.2 section 19 | raagacomposer/audio/export.py |
| 58 | First Complete Implementation | (v0.3 checklist; references all subsystems) | entire codebase |
| 59 | Working Milestone | (v0.3 milestone; implementation goal) | entire codebase |
| 60 | Reliable Milestone | (v0.3 milestone; implementation goal) | tests/ |
| 61 | Correct Milestone | (v0.3 milestone; implementation goal) | entire codebase |
| 62 | Excellent Milestone | (v0.3 milestone; implementation goal) | entire codebase |
| 63 | Key Acceptance Tests | Master Spec v0.2 section 21 | tests/integration/test_acceptance_scenario.py, tests/integration/test_agent_acceptance.py, tests/integration/test_training_acceptance.py |
| 64 | Immediate Implementation Order | (v0.3 roadmap; foundational) | docs/PLAN_v0.3.md |
| 65 | Hard-Blocker Policy | Master Spec v0.2 section 2 | docs/DECISIONS.md |
| 66 | Non-Goals for the First Build | (v0.3 scope-limiting) | docs/DECISIONS.md:534-537 |
| 67 | Final Directive | (v0.3 architectural vision) | raagacomposer/app.py, docs/DECISIONS.md |

---

## Summary

- **Total distinct old references mapped:** 47
- **Unmapped old references:** 0
- **v0.3 sections with explicit old mappings:** 51
- **v0.3 sections with no explicit old reference (foundational or new in v0.3):** 16
