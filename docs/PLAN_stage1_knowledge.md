# Stage 1 knowledge: the melakarta pack and explainable raga selection

Written 2026-09-04 against `docs/spec/stage1_knowledge_pack/` (Raaga Stage 1
Knowledge Pack v1.0), the canonical v0.3 specification and the Agent Factory
framework. The pack is treated as guidelines. Nothing below is built; the pack
is checked as an asset (`tests/unit/test_stage1_pack.py`) and this file is
the queue for using it.

## What the pack asks, and what exists

| Pack | Requirement | State today |
|---|---|---|
| 01 A | Swarasthana dictionary, 16 names on 12 pitch positions, overlaps kept as labels | `raaga/library.py` `SWARA_SEMITONES` agrees with the pack's pitch classes (tested) |
| 02 to 04 | 72 melakarta records, validated at startup | The library carries 18 raagas, 7 of them melakartas, all agreeing with the pack (tested); no 72-record table |
| 01 B to E, 05 section 2 | Block-character profiles: R-G block + M + D-N block, explainable | Raagas carry curated `moods`; no block model |
| 05 sections 1, 3, 5 | Brief to emotion vector; weighted scoring with contradiction penalties; diversity in the top five | `agent.suggest_raagas` ranks from the knowledge base and the library's moods with a deterministic fallback (v0.3 section 6, PR #4); no emotion vector, no penalties, no diversity rule |
| 05 section 6 | Selection feedback as learned weights, separate from grammar | Feedback lowers or raises phrase confidence and writes lessons (PRs #5, #10); no selection weights |
| 05 section 7, 06 E | Audition: play arohanam then avarohanam, eight events each, changing pitch | The practice engine renders scales for itself; no audition control in MAIN |
| 06 | Startup validation; mandatory tests A to E | A to C run against the pack files; D and E await the engine and the audition |

## Design, in brief

- **The 72 records join the library as hard knowledge**, loaded from the pack
  files at startup and validated by the pack's rules, with the existing 18
  entries taking precedence where they carry prayogas, nyasa and gamaka the
  pack does not have. A melakarta with no library entry composes from its
  scale alone, and the agent says so.
- **A block profile per melakarta** (R-G, M, D-N) with the pack's starter
  tags as `KnowledgeClass.HEURISTIC`, so a suggestion's reason is traceable to
  the map ("R2G2 tender, M1 grounded, D1N3 poignant") rather than to a name.
- **The selection engine** turns the brief into the pack's fourteen-dimension
  emotion target by a deterministic keyword dictionary (a provider-backed
  classifier when configured, per v0.3 section 41), scores with the pack's
  weights and penalties, and returns five with a spread. It replaces the
  ranking inside `suggest_raagas`, keeps its status contract, and never
  returns an empty list.
- **Selection feedback** becomes learned weights on the heuristic profile,
  stored beside the lessons and never touching grammar (pack 05 section 6,
  framework document 04 section 6).
- **Audition** is a MAIN control that plays the exact arohanam and
  avarohanam through the existing synthesiser, and a test that hears eight
  rising and eight falling events.

## Queue

| # | Task | Proves |
|---|---|---|
| S1 | Load and validate the 72 records; library precedence; a melakarta-only raaga composes from its scale | pack validation at startup; test C on live data |
| S2 | Block profiles and the emotion-vector scorer; explainable reasons; diversity | pack test D on the sad, romantic, lonely, warm brief; no single default raaga |
| S3 | Selection feedback as learned weights | a rejected suggestion ranks lower next time; grammar unchanged |
| S4 | Audition control and playback test | pack test E |

Each is one branch and one PR, verified live before it is offered.
