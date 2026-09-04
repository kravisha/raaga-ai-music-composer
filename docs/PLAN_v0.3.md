# Integrating Canonical Specification v0.3

The v0.3 specification (`docs/spec/CANONICAL_SPEC_v0.3.txt`, dated 2026-09-03)
consolidates the four earlier documents the code was written against: the
Master Specification v0.2, the Self-Learning Music Agent specification, the
Training-tab specification and the Knowledge Base architecture. Its section 64
gives an implementation order and its section 0 says to map requirements onto
what exists rather than build parallel systems. This file is the queue that
does that, kept in the section 65 status-report shape.

## What v0.3 changes

Inspection of the repository against v0.3 (section 64 step 1) found that most
of section 58's thirty-item end-to-end path already exists: the Music Agent,
Curriculum Engine, Knowledge Base that opens rather than recreates, provider
router, training search -> approval -> queue -> one-source processing ->
Learning Report -> KB with provenance, practice, evaluator, restart recovery,
project save/reopen. The items that are genuinely new or changed:

| v0.3 section | Requirement | State before this work |
|---|---|---|
| 4 | Two top-level workspaces, MAIN and LEARN | Learning and Training are two tabs beside Tune/Lyrics/Voice/Output |
| 6, 6.1 | Apply Brief must suggest raagas with visible status and never fail silently | "Apply brief" only saved the brief; suggestions needed a second button; no phase status |
| 5 | Brief includes a song title | Title lives on the project panel; brief has the other ten fields |
| 41 | Provider status visible: Configured / Not configured / Unavailable, local Available / Not installed / Ready | Only printed under Help -> About and in the output panel |
| 42 | Settings area to enter, validate, change and remove the Anthropic key; secure storage; redaction | Environment variable or hand-edited `credentials.json`; no UI; no redaction filter |
| 54 | Diagnostic export redacts secrets | Export bundles logs; nothing scrubs a key that reached a log line |
| 63 TEST A, G, H, I | Named acceptance tests | A, H, I have no test; G is covered by the melody golden files but not by name |

## Queue

Order follows section 64. Each item names the files it touches so two items
can run side by side only when they do not overlap.

| # | Task | Touches | Status |
|---|---|---|---|
| 1 | Bring v0.3 into the repo: the text, a cross-reference from old section numbers to v0.3, a DECISIONS.md entry, README pointer | `docs/spec/`, `docs/DECISIONS.md`, `README.md` | done |
| 2 | Fix Apply Brief and add the action status contract (section 6, 6.1, TEST A): Apply reads, validates, builds intent, queries the KB, ranks, shows phases, surfaces failure | `ui/panels/brief_panel.py`, `ui/panels/raaga_panel.py`, `app.py`, `core/actions.py`, `core/jobs.py` (epoch on the job context), `core/models.py` (title), new `tests/regression/test_apply_brief.py` | done |
| 3 | Secure Anthropic key configuration and provider status (sections 41, 42, 54, TEST H): keyring-backed secret store with file fallback, provider status model, settings dialog, log redaction, redacted diagnostics | new `core/secrets.py`, `core/settings.py`, new `providers/status.py`, new `ui/settings_dialog.py`, `core/logging_setup.py`, tests | done |
| 4 | MAIN and LEARN top-level workspaces (section 4, TEST I): LEARN gets Dashboard, Curriculum, Training Sources, Practice/Quiz, Knowledge, History areas from the existing widgets plus a new dashboard and practice view; MAIN keeps the composer; wire the settings dialog and provider status into the menu and status bar | `ui/main_window.py`, new `ui/learn_workspace.py`, `ui/panels/agent_panel.py`, `tests/integration/test_ui_window.py`, new `tests/integration/test_learn_workspace.py` | done |
| 5 | Acceptance tests by name: TEST A (Apply Brief), G (tune is multi-note), H (provider with and without key), I (LEARN is top-level); provider failure and fallback | `tests/regression/`, `tests/integration/` | done, see below |
| 6 | Verify live: launch the application, run Apply Brief, open the settings dialog, switch workspaces; run the full suite; commit; PR | - | verified live 2026-09-03; commit and PR pending the creator's word |

## Where each named acceptance test lives

| Section 63 test | Test function |
|---|---|
| A - Apply Brief | `tests/regression/test_apply_brief.py::test_apply_brief_sync_reports_progress_and_suggests_raagas` |
| B - persistence | `tests/integration/test_restart_recovery.py`, `tests/integration/test_kb_acceptance.py` (pre-existing) |
| C - training | `tests/integration/test_training_acceptance.py` (pre-existing) |
| D, E, F - duplicate, contradiction, correction | `tests/integration/test_kb_acceptance.py` (pre-existing) |
| G - tune | `tests/regression/test_generate_tune_acceptance.py::test_g_generate_tune_for_keeravani_is_real_playable_music` (controller path); `tests/integration/test_agent_acceptance.py::test_d_a_requested_prelude_is_real_music` (agent path) |
| H - provider | `tests/integration/test_provider_acceptance.py::test_provider_status_with_and_without_a_key`; step 5 (a live Claude call) is not automated because the suite never spends money |
| I - LEARN UI | `tests/integration/test_ui_window.py::test_learn_is_a_top_level_workspace_not_a_composition_tab` |
| J - crash | `tests/integration/test_restart_recovery.py` (pre-existing) |

## Provisional decisions taken

* **The v0.3 text is kept verbatim** under `docs/spec/` rather than rewritten
  as Markdown, so a section number quoted anywhere resolves to exactly the
  creator's wording. The cross-reference file is the bridge from the old
  numbering the code comments still use.
* **Code comments keep their old section numbers.** Sixty-seven references
  across the package point at v0.2 and agent-spec sections. Rewriting them is
  churn with no behavioural value; `docs/spec/CROSSREF.md` maps each to v0.3.
  New code cites v0.3 sections.
* **Apply Brief runs the suggestion, and the separate "Suggest from the brief"
  button stays.** Section 6 says Apply must produce ranked suggestions;
  section 7 says the creator can ask for alternatives. Both are satisfied by
  Apply doing the whole thing and the button re-running it.
* **Secure storage is `keyring` when it is installed, `credentials.json` when
  it is not.** Section 42 says "where practical". `keyring` installs cleanly
  on this machine and uses Windows Credential Manager; it goes in
  `requirements-optional.txt`, not `requirements.txt`, so the application
  still runs with nothing but the core dependencies. Environment variable
  still wins, as before, so `setx` keeps working.
* **Validation of a key is a live call only when the creator asks.** The
  dialog checks the shape of the key locally; "Validate" makes one cheap
  request. Nothing calls the network on startup.
* **LEARN is a top-level `QStackedWidget` page, not a second window.** A
  second window breaks the "one workstation" feel and makes the conversation
  dock ambiguous. A toolbar toggle switches between the two pages.

## Hard blockers

None.

## Tests / evidence

Run on 2026-09-03, Python 3.14.5, Windows 11, `pytest -q -p no:cacheprovider`:

| What | Result |
|---|---|
| Full suite before this work (main) | 913 passed |
| Full suite after items 1-4 | 955 passed, 7 min 36 s |
| `tests/regression/test_generate_tune_acceptance.py` (added after that run) | 1 passed |
| `tests/regression/test_apply_brief.py` | 9 passed |
| Secrets, redaction, provider status, settings dialog, TEST H, routing | 117 passed |
| UI window, LEARN workspace, agent-in-app, training flow and acceptance | 81 passed |

Live, against a scratch config directory (`RAAGA_COMPOSER_HOME` pointed at a
temporary folder, file secret backend, so the creator's own settings,
credential store and databases were not touched): the application launched;
the composer tab bar showed Tune / Lyrics / Voice / Output only; MAIN and
LEARN switched from the toolbar and from Ctrl+1 / Ctrl+2; LEARN listed its
six areas with a populated dashboard (stage A, Keeravani, unit a01.sound,
mastery 0.45); Ctrl+, opened Settings with the masked key field, "Stored in:
Not configured", and the provider table (two Claude rows Not configured, four
local rows Not installed); typing "love failure" and "lonely late at night
but still warm" and pressing Apply brief produced, in the log and on screen,
"Analyzing creative brief...", "Searching learned raga knowledge...",
"Ranking suggestions..." and "4 raagas suggested; Keeravani first." with
four ranked raagas and reasons in the Raaga panel. Screenshots were handed to
the creator.

Not verified: a live Claude call with a real key (TEST H step 5), storage
in Windows Credential Manager with a real key (the file backend was forced),
and any speech backend, since none is installed.

One pre-existing test, `test_reg_095_research_does_not_spin_when_material_runs_out`,
failed in three of five full-suite runs on 2026-09-03 and passed alone every
time. It was handled as separate work: the practice seed was the wall-clock
second, so every retry made within one second replayed the failed attempt
note for note until the twelve-attempt cap. Fixed on `main` in PR #3
(seed from the unit id and attempt number, REG-100); this branch is rebased
on that fix and the full suite passed 958 on 2026-09-04.

A second one-off, `test_the_queue_worker_processes_everything_it_is_given`,
stalled once in a full-suite run on 2026-09-04 (commit c3f87e6): "training
queue started", then nothing for its 120-second deadline, both runs still
queued, no exception in any thread. The cause was the training store sharing
one sqlite3 connection between the queue worker and the UI thread with
nothing serialising them - the same defect the agent's KnowledgeRepository
had been given a lock for in PR #5, and which sqlite3 turns into
`InterfaceError: bad parameter or other API misuse`, a cursor reset under
another thread's feet, or a worker that never returns, depending on the
interleaving. The worker's `_next()` and the test's `pending()` poll run the
same SELECT, so they contended for one cached statement; polling every 100 ms
made it rare, polling flat out reproduces it every time. `TrainingStore`
now holds an `RLock` for every statement, with a unit test
(`test_training_store_threads.py`) and a queue-level regression
(`test_reg_the_queue_worker_must_survive_the_ui_polling_the_store`) that
both fail on the old store within two seconds.

## Next milestone

Section 59 (WORKING): Apply Brief meaningfully ranks raagas from the Knowledge
Base with the Claude route when configured; LEARN is a separate screen while
MAIN remains usable.
