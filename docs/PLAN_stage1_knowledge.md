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
| 02 to 04 | 72 melakarta records, validated at startup | All 72 are in the library, generated from the pack and validated when generated and again by test (S1) |
| 01 B to E, 05 section 2 | Block-character profiles: R-G block + M + D-N block, explainable | Every melakarta carries its blocks and their character, rendered by `block_summary()` and used in the reason a suggestion gives (S1); no scoring from them yet |
| 05 sections 1, 3, 5 | Brief to emotion vector; weighted scoring with contradiction penalties; diversity in the top five | Built (S2): `raaga/emotion.py`, used by both ranking paths. A semantic classifier for the target vector is still to come; the deterministic reader is the fallback it will fall back to |
| 05 section 6 | Selection feedback as learned weights, separate from grammar | Feedback lowers or raises phrase confidence and writes lessons (PRs #5, #10); no selection weights |
| 05 section 7, 06 E | Audition: play arohanam then avarohanam, eight events each, changing pitch | The practice engine renders scales for itself; no audition control in MAIN |
| 06 | Startup validation; mandatory tests A to E | A to C run against the pack files, C also against the live library; D and E await the engine and the audition |

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

| # | Task | Proves | Status |
|---|---|---|---|
| S1 | Load and validate the 72 records; library precedence; a melakarta-only raaga composes from its scale | pack validation at startup; test C on live data | done |
| S2 | Block profiles and the emotion-vector scorer; explainable reasons; diversity | pack test D on the sad, romantic, lonely, warm brief; no single default raaga | done |
| S3 | Selection feedback as learned weights | a rejected suggestion ranks lower next time; grammar unchanged | open |
| S4 | Audition control and playback test | pack test E | open |

Each is one branch and one PR, verified live before it is offered.

## S1, as built

`tools/build_melakartas.py` generates `raagacomposer/raaga/data/melakartas.json`
from the pack and validates every record against the pack's own rules;
`RaagaLibrary.load_melakartas` merges it in, joining on the melakarta number
rather than on any name. The library holds 82 raagas: the 18 curated ones (8
of them melakartas, which keep everything they had and gain the chakra, the
three blocks, the block characters and the starter tags) plus 64 melakartas
that arrive as a scale and its character and say so. The judgment calls are
in `docs/DECISIONS.md` under "Stage 1 knowledge pack".

What S2 inherited: `Raaga.rg`, `Raaga.madhyama`, `Raaga.dn`,
`Raaga.block_character`, `Raaga.tags` and `Raaga.good_for` on all 72, with
`block_summary()` already rendering the explainable sentence.

## S2, as built

`raagacomposer/raaga/emotion.py` is the pack's engine: `target_vector` reads
the brief into the fourteen dimensions, `profile_vector` reads a raaga out of
its blocks and its curation, `score_raaga` compares them by cosine and
applies the pack's contradiction penalties and block bonuses, and `spread`
picks a diverse five without reordering them. It replaces the ranking inside
both `raaga/selection.suggest` and `MusicAgent.suggest_raagas`, which keep
their shapes, their status contract and their never-empty guarantee.

Pack test D is
`tests/unit/test_emotion_selection.py::test_d_brief_selection_smoke_test`.
The judgment calls are in `docs/DECISIONS.md` under "Stage 1 knowledge pack".

What S3 inherits: scoring is one function of a target vector and a profile
vector, so a learned weight is a multiplier on a dimension or on a block's
contribution and never touches `arohanam`, `avarohanam` or anything else the
pack marks `[GRAMMAR]` - which is what pack document 05 section 6 and
framework document 04 section 6 both require. `Scored` already carries the
fit, the bonuses and the penalties a feedback signal would adjust, and
`Raaga.tags` is the per-raaga heuristic the weights would attach to.
