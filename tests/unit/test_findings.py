"""Unit tests: structured findings alongside the evaluator's prose mistakes.

Learning specification section 26 ("detected mistakes") and section 38 (the
Failure/Lesson object needs a kind and evidence to work from).  Every finding
is checked against the prose it was built from too, so the two never drift
apart.
"""
from __future__ import annotations

import pytest

from raagacomposer.agent.evaluator import Evaluation, Evaluator, Finding
from raagacomposer.agent.knowledge import Phrase
from raagacomposer.agent.originality import PhraseIndex, check
from raagacomposer.core.models import Note

pytestmark = pytest.mark.unit

TONIC = 60


def notes(raaga, swaras, duration=0.5, gamaka=False, velocity=90):
    out = []
    t = 0.0
    for swara in swaras:
        out.append(Note(swara=swara, midi=raaga.midi(swara, TONIC), start=t,
                        duration=duration, velocity=velocity,
                        gamaka=raaga.gamaka_for(swara) if gamaka else ""))
        t += duration
    return out


def _findings(evaluation: Evaluation, kind: str):
    return [f for f in evaluation.findings if f.kind == kind]


# --------------------------------------------------------------------------
def test_nothing_produced_finding(raagas, keeravani):
    evaluation = Evaluator(raagas).evaluate([], keeravani)
    found = _findings(evaluation, "nothing_produced")
    assert len(found) == 1
    assert found[0].dimension == "structure"
    assert found[0].text == "nothing was produced"


def test_outside_swara_finding(raagas, keeravani):
    wrong = notes(keeravani, ["S", "R2", "G2", "M1"])
    wrong[2].swara = "G3"
    wrong[2].midi = TONIC + 4
    evaluation = Evaluator(raagas).evaluate(wrong, keeravani)
    found = _findings(evaluation, "outside_swara")
    assert len(found) == 1
    assert found[0].dimension == "swara_correctness"
    assert found[0].evidence == "G3"


def test_no_cadence_finding(raagas, keeravani):
    restless = notes(keeravani, ["S", "R2", "G2", "M1", "D1"])
    evaluation = Evaluator(raagas).evaluate(restless, keeravani)
    found = _findings(evaluation, "no_cadence")
    assert len(found) == 1
    assert found[0].dimension == "structure"
    assert found[0].evidence == "D1"       # the actual final swara


def test_too_many_leaps_finding(raagas, keeravani):
    jumpy = notes(keeravani, ["S", "P", "S-", "N3", "S", "D1-"])
    evaluation = Evaluator(raagas).evaluate(jumpy, keeravani)
    found = _findings(evaluation, "too_many_leaps")
    assert len(found) == 1
    assert found[0].dimension == "coherence"
    assert found[0].evidence.isdigit()
    assert int(found[0].evidence) > 0


def test_repetitive_finding(raagas, keeravani):
    dull = notes(keeravani, ["G2"] * 6)
    evaluation = Evaluator(raagas).evaluate(dull, keeravani)
    found = _findings(evaluation, "repetitive")
    assert len(found) == 1
    assert found[0].dimension == "interest"


def test_not_original_finding(raagas, keeravani):
    idiom = ["S", "R2", "G2", "M1", "P", "D1", "N3"]
    index = PhraseIndex(n=4)
    index.add(Phrase(raaga="Keeravani", swaras=idiom))
    evaluation = Evaluator(raagas, index).evaluate(notes(keeravani, idiom),
                                                   keeravani)
    found = _findings(evaluation, "not_original")
    assert len(found) == 1
    assert found[0].dimension == "originality"
    assert found[0].weight == pytest.approx(1.5)
    assert evaluation.originality.matched_phrase_id in found[0].evidence
    assert str(evaluation.originality.longest_match) in found[0].evidence


# --------------------------------------------------------------------------
def test_findings_and_mistakes_are_in_lockstep(raagas, keeravani):
    """Every finding's text is also a mistake, same count, same order - the
    prose reader and the structured reader never see a different story."""
    wrong = notes(keeravani, ["S", "R2", "G2", "M1", "D1"])
    wrong[2].swara = "G3"
    wrong[2].midi = TONIC + 4
    evaluation = Evaluator(raagas).evaluate(wrong, keeravani)
    assert evaluation.findings
    assert [f.text for f in evaluation.findings] == evaluation.mistakes


def test_the_report_still_lists_problems(raagas, keeravani):
    wrong = notes(keeravani, ["S", "R2", "G2", "M1"])
    wrong[2].swara = "G3"
    wrong[2].midi = TONIC + 4
    evaluation = Evaluator(raagas).evaluate(wrong, keeravani)
    report = evaluation.report()
    assert "problems:" in report
    for mistake in evaluation.mistakes:
        assert mistake in report


def test_evaluation_note_appends_both(raagas, keeravani):
    evaluation = Evaluation()
    evaluation.note("structure", "no_cadence", "it does not rest", evidence="D1",
                    weight=1.2)
    assert evaluation.mistakes == ["it does not rest"]
    assert len(evaluation.findings) == 1
    finding = evaluation.findings[0]
    assert isinstance(finding, Finding)
    assert (finding.dimension, finding.kind, finding.evidence, finding.weight) == \
        ("structure", "no_cadence", "D1", 1.2)
