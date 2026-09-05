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
| 05 section 6 | Selection feedback as learned weights, separate from grammar | Built (S3): `selection_weights` in knowledge.db, per raaga and per emotion dimension, read only by the ranking |
| 05 section 7, 06 E | Audition: play arohanam then avarohanam, eight events each, changing pitch | Built (S4): `raaga/audition.py`, `AppController.audition_raaga`, "Hear the scale" in the raaga panel |
| 06 | Startup validation; mandatory tests A to E | All five run: A to C against the pack files and the live library, D in `test_emotion_selection.py`, E in `test_audition.py` |

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
| S3 | Selection feedback as learned weights | a rejected suggestion ranks lower next time; grammar unchanged | done |
| S4 | Audition control and playback test | pack test E | done |

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

## S3, as built

`selection_weights` in `knowledge.db` (schema 3) stores what the creator's
choices taught us as `(raaga, dimension, weight, observations)`: one row per
emotion dimension the brief was asking for, plus a `"*"` row for the raaga
however it was asked for. `KnowledgeRepository.record_selection_feedback`
writes it, `selection_weight_map` reads it in one go for a whole ranking, and
`emotion.feedback_bias` turns it into a bounded adjustment that counts each
dimension in proportion to how much *this* brief is asking for it - which is
what keeps a rejection in a joyful brief from sinking the same raaga in a
grieving one.

Signals are the pack's: accepted +1.0, auditioned +0.2, rejected -0.7, plus
passed-over -0.25. `AppController.select_raaga` sends them when a creator
chooses (and `require_raaga`'s automatic pick explicitly does not),
`AppController.reject_raaga` sends a rejection and reads any comment through
`emotion.read_correction`. `MusicAgent.selection_preferences` and
`forget_selection_preferences` are the review and reset.

The judgment calls are in `docs/DECISIONS.md` under "Stage 1 knowledge pack".

What S4 inherits: `MusicAgent.audition_raaga` already exists and sends the
pack's +0.2 signal; the audition control has only to call it after playing
the arohanam and avarohanam (pack document 05 section 7 steps C to E, and
document 06's test E).

## Stage 1, complete

All four queue items are built and the pack's five mandatory tests are named
in the suite:

| Pack test | Where |
|---|---|
| A - Keeravani is #21 | `tests/unit/test_stage1_pack.py::test_a_keeravani_is_number_21` |
| B - swara overlaps | `tests/unit/test_stage1_pack.py::test_b_swara_overlaps_share_a_pitch_but_keep_their_labels` |
| C - melakarta endpoints | `test_c_melakarta_endpoints`, and `test_pack_test_c_endpoints_are_live_library_entries` against the loaded library |
| D - brief selection | `tests/unit/test_emotion_selection.py::test_d_brief_selection_smoke_test` |
| E - playback | `tests/unit/test_audition.py::test_e_playback_smoke_test` |

What Stage 1 does not claim: that a parent scale is a lived raga.  The pack
says so itself, and the application says so too - a melakarta nobody curated
is offered at lower confidence, describes itself as a scale, and composes
from that scale alone.  Gamaka, characteristic prayoga, nyasa and phrase
grammar remain the agent's to learn.

Open afterwards, and not Stage 1's business: the calibration numbers this
work chose rather than measured - `LEARNED_BONUS_CAP`, the selection signal
strengths, and the diversity weight - all of which now have the machinery to
be settled from real use.
