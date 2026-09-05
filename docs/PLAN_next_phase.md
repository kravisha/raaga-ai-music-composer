# Next phase — the two specifications, mapped

Sources: `Raaga_AI_Next_Phase_Spec_2026-09-05.txt` and
`Raaga_AI_Jam_Studio_Excellence_Addendum_2026-09-05.txt`.

The addendum's §23 is explicit that Jam Studio must not interrupt the staged
correctness work, so the order below is the spec's, with the addendum
captured now because it changes *how* several of those stages should be
built — not when.

## Already delivered

| § | where |
|---|---|
| 3 Octave correctness | `analysis.repair_octaves`, `widest_leap`; PR #25 |
| 17 Versioning, rebuilding, traceability | `ANALYSIS_VERSION`, `tools/rebuild_knowledge.py`, `KnowledgeRepository.forget_source`; PR #26 |
| 2 Verification gate | run as process: thresholds fixed before the run, one sample verified before bulk |
| 1 Persistence (most of it) | `sources`, `phrases`, `raaga_facts`, the KB bridge with evidence and extraction version |
| 15 Microphone crash | this PR |

Still missing from §1: a lifecycle status for a learned item (active,
disputed, superseded, deprecated) and relationships between constructions.

## The stages

**1 — Voice (§15).** Done here. The crash, the visible phases, graceful
failure.

**2 — Cheap, visible clarity.** §12 (a generated tune states its target
raaga, the detected raaga, confidence and any out-of-raaga notes — the
evaluator already computes all of it), §13 (which instrument you are
hearing and why), §14 (rename "Suggest from the brief", see below).

**3 — Readable raaga details (§11, §16).** Double-click a suggestion for a
spacious, sectioned, resizable dialog.

**4 — The knowledge model (§4, §7, §1's remainder).** Confidence that moves
rather than facts that are deleted; supersession and retraction told apart
from source removal; constructions of 10–20 notes rather than the current
2–10.

**5 — Mastery and quality (§5, §6, §8).** Per-raaga mastery profiles, a
generated versioned golden set, and impact signals stored apart from
correctness.

**6 — Jam Studio (addendum, phases B–F).** Only after the above.

## What the addendum changes about stage 4 and 5

This is why it is worth reading now rather than at phase B.

**`Evaluation.overall()` is the thing the addendum forbids.** Twelve
dimensions collapse into one weighted scalar:

```
correctness   swara_correctness, raaga_correctness, raaga_drift, phrase_authenticity
quality       interest, expressiveness, coherence, structure, originality
fit           mood_match, brief_match
craft         rhythm
```

Addendum §7: *"Do not assume these can all be reduced to one scalar
score."* Spec §8 says the same of correctness versus musical quality. This
is not theoretical — Hindolam's `phrase_authenticity` rose 0.768 → 0.942
after the agent heard 124 phrases, and `overall` moved 0.737 → 0.753,
because authenticity is one dimension of twelve. **The learning worked and
the score hid it.**

So stage 5 should split the evaluation into named groups that stay separate
all the way to the UI, rather than adding a quality layer beside a scalar
that already averages quality away.

**Preference learning already exists and is already fragile.**
`selection_weights` learns which raaga suits which feeling from what the
creator accepts, auditions and passes over. The addendum's §6 and §9 extend
exactly this with pairwise and ranked preferences from jam sessions.

The warning from building it: on 2026-09-05 the audition replayed the
previous raaga's scale, so several rows recorded a judgement of one raaga
formed while hearing another. **Preference learning must record what was
actually rendered, not what was requested** — otherwise a UI bug becomes a
permanent false belief, and unlike machine-derived knowledge it cannot be
re-derived.

**Constructions are the same requirement in both documents.** Spec §7 wants
musically meaningful units of roughly 10–20 notes; addendum §5 wants
anything from a motif to an alapana. Today `learned_phrase_bank` caps at 10
swaras and `_extract` requires at least 3 distinct. One change serves both.

**Already built for the addendum:** `agent/originality.py`
(`check_originality`, `PhraseIndex`) is §18 — the agent already refuses to
quote a learned phrase verbatim. `speech/` and the conversation panel are
the input modes of §4. `factory.db`'s `mastery` table is the seed of §20's
Excellence Profile, which should sit *beside* the mastery profile, not
inside it: mastery answers "does it understand this raaga", excellence
answers "can it write something compelling in it".

## §14 — "Suggest from the brief", as the spec asks, before touching it

`raaga_panel.py` → `app.raaga_suggestions()` → `_run_apply_brief_pipeline`,
which is **the same pipeline Apply brief runs**.

* **Apply brief** reads the form fields, writes them into the project
  brief, validates that situation/mood/feel say something, then ranks.
* **Suggest from the brief** ignores the form, uses the stored brief, skips
  validation, and re-ranks.

They overlap almost entirely, and the raaga panel already refreshes itself
when `apply_brief` completes, so the suggestions appear without the button
being pressed at all.

Its one real use is re-ranking after the agent has learned something, since
`selection_weights` changes the order without the brief changing.

**Recommendation: rename to "Re-rank with what I've learned". Do not
remove.** The behaviour is real; only the label duplicates the other button.
