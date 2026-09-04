"""Unit tests: the practice engine obeying guidance (docs/PLAN_learning_loop.md,
"Generation obeys guidance").

Two things have to be true at once, and this file is organised around them:

* With no guidance at all (``None`` or an empty ``Guidance()``) every
  generator makes exactly the same random draws and produces exactly the
  same tokens, notes and scores as before guidance existed - the "Determinism
  stays" decision, and the hard constraint this whole item was built under.
* With guidance, the constraint it names actually holds over many seeds, not
  just the one it was tuned against.
"""
from __future__ import annotations

import random
import statistics

import pytest

from raagacomposer.agent.curriculum import CurriculumEngine
from raagacomposer.agent.guidance import Guidance, guidance_from_lessons
from raagacomposer.agent.knowledge import KnowledgeRepository, Lesson, Phrase
from raagacomposer.agent.originality import PhraseIndex, check as check_originality
from raagacomposer.agent.practice import PracticeEngine
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.unit

SEEDS_30 = range(1, 31)
SEEDS_40 = range(1, 41)


@pytest.fixture
def repo(tmp_path) -> KnowledgeRepository:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


@pytest.fixture
def engine(repo, raagas, settings) -> PracticeEngine:
    return PracticeEngine(repo, raagas, settings)


def _teach(repo: KnowledgeRepository, raaga_name: str, phrases) -> None:
    for swaras in phrases:
        repo.add_phrase(Phrase(raaga=raaga_name, swaras=list(swaras),
                               confidence=0.8))


def _lessons_from_findings(report, raaga: str, unit_id: str):
    """The same collapsing rule as ``MusicAgent.record_lessons`` - one lesson
    per finding kind - but as plain objects, with no repository write."""
    seen = set()
    lessons = []
    for finding in report.findings:
        if finding.kind in seen:
            continue
        seen.add(finding.kind)
        related = []
        if (finding.kind == "not_original" and report.evaluation is not None
                and report.evaluation.originality
                and report.evaluation.originality.matched_phrase_id):
            related = [report.evaluation.originality.matched_phrase_id]
        lessons.append(Lesson(raaga=raaga, unit_id=unit_id, kind=finding.kind,
                              dimension=finding.dimension,
                              failure_reason=finding.text,
                              evidence=finding.evidence, related=related))
    return lessons


# ==========================================================================
# Unguided parity: guidance=None and guidance=Guidance() are the same code
# path, and the empty case reproduces today's output exactly.
# ==========================================================================
def test_motif_is_unchanged_by_an_empty_guidance(engine, keeravani):
    bank = [list(p) for p in keeravani.prayogas]
    for seed in SEEDS_30:
        without = engine._motif(keeravani, random.Random(seed), 6, bank, True)
        with_empty = engine._motif(keeravani, random.Random(seed), 6, bank,
                                   True, Guidance())
        with_none = engine._motif(keeravani, random.Random(seed), 6, bank,
                                  True, None)
        assert without == with_empty == with_none, seed


def test_notes_from_tokens_is_unchanged_by_an_empty_guidance(engine, keeravani):
    tokens = engine._motif(keeravani, random.Random(5), 6,
                           [list(p) for p in keeravani.prayogas], True)
    for seed in SEEDS_30:
        without = engine._notes_from_tokens(keeravani, tokens,
                                            random.Random(seed), 76.0)
        with_empty = engine._notes_from_tokens(keeravani, tokens,
                                               random.Random(seed), 76.0,
                                               Guidance())
        with_none = engine._notes_from_tokens(keeravani, tokens,
                                              random.Random(seed), 76.0, None)
        for a, b, c in zip(without, with_empty, with_none):
            assert (a.swara, a.midi, a.start, a.duration, a.velocity,
                    a.gamaka) == \
                   (b.swara, b.midi, b.start, b.duration, b.velocity, b.gamaka)
            assert (a.swara, a.midi, a.start, a.duration, a.velocity,
                    a.gamaka) == \
                   (c.swara, c.midi, c.start, c.duration, c.velocity, c.gamaka)


def test_a_full_run_is_unchanged_by_an_empty_guidance(repo, raagas, settings,
                                                       keeravani):
    _teach(repo, "Keeravani", keeravani.prayogas)
    engine = PracticeEngine(repo, raagas, settings)
    unit = CurriculumEngine(repo).unit("b13.short_phrase:Keeravani")
    for seed in SEEDS_30:
        none_report = engine.run(unit, "Keeravani", seed=seed, guidance=None)
        empty_report = engine.run(unit, "Keeravani", seed=seed,
                                  guidance=Guidance())
        assert none_report.score == empty_report.score, seed
        assert [e.heard for e in none_report.exercises] == \
            [e.heard for e in empty_report.exercises], seed


# ==========================================================================
# Structural invariants: what each constraint promises actually holds over
# many seeds, not only the one it was written against.
# ==========================================================================
def test_must_end_on_nyasa_forces_a_resting_ending(engine, keeravani):
    guidance = Guidance(must_end_on_nyasa=True)
    nyasa = set(keeravani.nyasa)
    for seed in SEEDS_40:
        # length 3, cadence=False: without the guidance flag nothing would
        # force an ending at all, so this isolates the flag itself.
        tokens = engine._motif(keeravani, random.Random(seed), 3, None,
                               False, guidance)
        assert tokens and parse_swara(tokens[-1])[0] in nyasa, (seed, tokens)


def test_avoid_transitions_never_appears(engine, keeravani):
    # Chosen empirically as a transition the unguided generator actually
    # makes across these seeds (docs/PLAN_learning_loop.md item 2's own
    # verification requirement), with a destination that is not one of the
    # raaga's resting notes - the cadence step chooses only among
    # ``avoid_endings``-filtered nyasa swaras and does not re-check
    # ``avoid_transitions``, so a transition landing on a nyasa swara could
    # still appear there and would not be a fair pick for this invariant.
    avoided = ("G2", "R2")
    assert not (avoided[1] in set(keeravani.nyasa))
    made_unguided = set()
    for seed in SEEDS_40:
        tokens = engine._motif(keeravani, random.Random(seed), 6, None, True)
        made_unguided.update(
            (parse_swara(a)[0], parse_swara(b)[0])
            for a, b in zip(tokens, tokens[1:]))
    assert avoided in made_unguided, \
        "the transition chosen for this test never actually occurs"

    guidance = Guidance(avoid_transitions={avoided})
    for seed in SEEDS_40:
        tokens = engine._motif(keeravani, random.Random(seed), 6, None, True,
                               guidance)
        made = {(parse_swara(a)[0], parse_swara(b)[0])
                for a, b in zip(tokens, tokens[1:])}
        assert avoided not in made, (seed, tokens)


def test_avoid_swaras_never_appears(engine, keeravani):
    # R2 is not one of the raaga's graha swaras, so it cannot sneak in
    # through the ungated initial choice of ``current`` either.
    assert "R2" not in set(keeravani.graha)
    guidance = Guidance(avoid_swaras={"R2"})
    for seed in SEEDS_40:
        tokens = engine._motif(keeravani, random.Random(seed), 6, None, True,
                               guidance)
        assert "R2" not in {parse_swara(t)[0] for t in tokens}, (seed, tokens)


def test_avoid_endings_never_ends_the_line(engine, keeravani):
    avoided = "G2"
    assert avoided in set(keeravani.nyasa)   # a real candidate, not a no-op
    guidance = Guidance(avoid_endings={avoided})
    for seed in SEEDS_40:
        tokens = engine._motif(keeravani, random.Random(seed), 6, None, True,
                               guidance)
        assert parse_swara(tokens[-1])[0] != avoided, (seed, tokens)


def test_avoid_quoting_keeps_the_line_original(repo, raagas, settings,
                                               keeravani):
    # Deliberately not the raaga's own prayogas: those are plain scale runs,
    # and Keeravani's arohanam is that same run, so a purely stepwise walk
    # reproduces one by chance often enough to make the invariant flaky for
    # reasons that have nothing to do with quoting.  These phrases contain
    # jumps of more than the generator's own two-scale-degree leap budget, so
    # only deliberate quoting - which ``avoid_quoting`` forbids here - could
    # ever reproduce five notes of one in a row.
    bank = [
        ["S", "P", "R2", "N3", "D1", "G2"],
        ["N3", "S", "M1", "D1", "R2", "P"],
        ["G2", "N3", "S", "P", "M1", "D1"],
    ]
    _teach(repo, "Keeravani", bank)
    engine = PracticeEngine(repo, raagas, settings)
    unit = CurriculumEngine(repo).unit("b13.short_phrase:Keeravani")
    index = PhraseIndex.from_repository(repo, "Keeravani")
    all_ids = {p.id for p in repo.phrases(raaga="Keeravani", limit=200)}
    assert all_ids
    guidance = Guidance(avoid_quoting=all_ids)
    for seed in SEEDS_40:
        report = engine.run(unit, "Keeravani", seed=seed, guidance=guidance)
        for notes in report.artifacts:
            swaras = [n.swara for n in notes]
            assert check_originality(swaras, index).is_original, (seed, swaras)


def test_avoid_runs_never_replays_three_notes_of_a_bank_phrase(engine,
                                                                keeravani):
    # Built only from swaras that are not resting notes (nyasa = S, P, G2 in
    # Keeravani): the cadence step at the end of ``_motif`` always overwrites
    # the last token with a nyasa swara and is not guarded by ``replays()``
    # (only the free walk and the quoting branch are, per
    # docs/PLAN_learning_loop.md's own wiring), so a run that could only ever
    # be replayed by landing on a nyasa ending would make this invariant
    # depend on a code path this test is not exercising.  With every note in
    # every avoided run drawn from {R2, M1, D1, N3}, the forced nyasa ending
    # can never complete a match, so what remains under test is exactly the
    # two guarded code paths.
    bank = [
        ["R2", "D1", "M1", "N3", "R2"],
        ["N3", "M1", "D1", "R2", "N3"],
        ["D1", "R2", "N3", "M1", "D1"],
    ]
    avoid_runs = {tuple(p) for p in bank}
    guidance = Guidance(avoid_runs=avoid_runs)

    def replayed(tokens):
        bases = [parse_swara(t)[0] for t in tokens]
        for i in range(len(bases) - 2):
            window = tuple(bases[i:i + 3])
            for run in avoid_runs:
                for start in range(len(run) - 2):
                    if run[start:start + 3] == window:
                        return window
        return None

    for seed in SEEDS_40:
        tokens = engine._motif(keeravani, random.Random(seed), 6, bank, True,
                               guidance)
        assert replayed(tokens) is None, (seed, tokens)


def test_unguided_parity_still_holds_with_avoid_runs_wired_in(engine,
                                                               keeravani):
    """Guards against the avoid_runs wiring itself: with no guidance at all
    (avoid_runs empty either way) the two new checks are no-ops and the line
    is exactly what it was before this lever existed."""
    bank = [list(p) for p in keeravani.prayogas]
    for seed in SEEDS_30:
        without = engine._motif(keeravani, random.Random(seed), 6, bank, True)
        with_empty = engine._motif(keeravani, random.Random(seed), 6, bank,
                                   True, Guidance())
        assert without == with_empty, seed


# ==========================================================================
# add_gamaka
# ==========================================================================
def test_add_gamaka_gives_an_otherwise_bare_line_at_least_one(engine,
                                                               keeravani):
    from dataclasses import replace
    # No jeeva/nyasa to trigger the "expressive" bonus and no gamaka table
    # to draw from, at a tempo fast enough that no note is long enough to
    # earn the >= 0.7s fallback either: nothing here would ever get an
    # ornament without the guidance flag.
    bare = replace(keeravani, gamaka={}, jeeva=[], nyasa=[])
    tokens = ["R2", "G2", "M1", "P", "D1"]
    unguided = engine._notes_from_tokens(bare, tokens, random.Random(1), 300.0)
    assert all(not n.gamaka for n in unguided)

    guided = engine._notes_from_tokens(bare, tokens, random.Random(1), 300.0,
                                       Guidance(add_gamaka=True))
    assert any(n.gamaka for n in guided)


# ==========================================================================
# Property statistic: guidance built from a failed attempt's own findings
# should not make the retry worse, on average.
# ==========================================================================
@pytest.mark.slow
def test_guided_retries_do_not_score_worse_on_average(repo, raagas, settings,
                                                       keeravani, capsys):
    _teach(repo, "Keeravani", keeravani.prayogas)
    engine = PracticeEngine(repo, raagas, settings)
    unit = CurriculumEngine(repo).unit("b13.short_phrase:Keeravani")

    unguided_scores = []
    guided_scores = []
    for seed in range(1, 41):
        unguided = engine.run(unit, "Keeravani", seed=seed, guidance=None)
        unguided_scores.append(unguided.score)
        lessons = _lessons_from_findings(unguided, "Keeravani", unit.id)
        guidance = guidance_from_lessons(lessons, unit_id=unit.id)
        guided = engine.run(unit, "Keeravani", seed=seed, guidance=guidance)
        guided_scores.append(guided.score)

    mean_unguided = statistics.fmean(unguided_scores)
    mean_guided = statistics.fmean(guided_scores)
    print(f"\nmean unguided score: {mean_unguided:.4f}")
    print(f"mean guided score:   {mean_guided:.4f}")
    assert mean_guided >= mean_unguided
