"""Unit tests: the critic and the originality safeguards."""
from __future__ import annotations

import pytest

from raagacomposer.agent.evaluator import DIMENSIONS, Evaluation, Evaluator
from raagacomposer.agent.knowledge import KnowledgeRepository, Phrase
from raagacomposer.agent.originality import (PhraseIndex, check, most_similar)
from raagacomposer.core.models import CreativeBrief, Note

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


@pytest.fixture
def keeravani_notes(keeravani):
    return notes(keeravani, ["S", "R2", "G2", "M1", "P", "M1", "G2", "S"])


# --------------------------------------------------------------------------
# originality
# --------------------------------------------------------------------------
def test_an_empty_index_calls_everything_original():
    report = check(["S", "R2", "G2", "M1"], PhraseIndex())
    assert report.is_original
    assert "nothing learned" in report.reason


def test_a_copied_line_is_caught():
    index = PhraseIndex(n=4)
    index.add(Phrase(raaga="Keeravani",
                     swaras=["S", "R2", "G2", "M1", "P", "D1", "N3"]))
    report = check(["S", "R2", "G2", "M1", "P", "D1", "N3"], index, max_run=5)
    assert not report.is_original
    assert report.longest_match >= 7
    assert "match" in report.summary()


def test_a_short_quotation_is_allowed():
    index = PhraseIndex(n=4)
    index.add(Phrase(raaga="Keeravani",
                     swaras=["S", "R2", "G2", "M1", "P", "D1", "N3", "S+"]))
    report = check(["P", "M1", "G2", "R2", "S", "R2", "G2", "M1"], index,
                   max_run=6)
    assert report.is_original
    assert report.score > 0.4


def test_copying_across_octaves_is_still_copying():
    index = PhraseIndex(n=4)
    index.add(Phrase(raaga="Keeravani", swaras=["S", "R2", "G2", "M1", "P"]))
    report = check(["S+", "R2+", "G2+", "M1+", "P+"], index, max_run=4)
    assert not report.is_original


def test_the_index_can_be_built_from_memory(tmp_path):
    repo = KnowledgeRepository(tmp_path / "k.db")
    try:
        repo.add_phrase(Phrase(raaga="Keeravani",
                               swaras=["S", "R2", "G2", "M1", "P"],
                               confidence=0.8))
        index = PhraseIndex.from_repository(repo, "Keeravani", n=4)
        assert index.size == 1
        assert most_similar(["S", "R2", "G2", "M1", "P"], index) is not None
    finally:
        repo.close()


# --------------------------------------------------------------------------
# the critic
# --------------------------------------------------------------------------
def test_every_dimension_is_scored_separately(raagas, keeravani,
                                              keeravani_notes):
    evaluation = Evaluator(raagas).evaluate(keeravani_notes, keeravani,
                                            tempo_bpm=72)
    assert set(evaluation.scores) == set(DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in evaluation.scores.values())
    assert 0.0 <= evaluation.overall() <= 1.0
    assert "overall" in evaluation.report()


def test_a_line_inside_the_raaga_scores_well(raagas, keeravani,
                                             keeravani_notes):
    evaluation = Evaluator(raagas).evaluate(keeravani_notes, keeravani,
                                            tempo_bpm=72)
    assert evaluation.scores["swara_correctness"] == 1.0
    assert evaluation.scores["raaga_correctness"] > 0.9


def test_a_foreign_note_is_caught(raagas, keeravani):
    wrong = notes(keeravani, ["S", "R2", "G2", "M1"])
    wrong[2].swara = "G3"
    wrong[2].midi = TONIC + 4
    evaluation = Evaluator(raagas).evaluate(wrong, keeravani)
    assert evaluation.scores["swara_correctness"] < 1.0
    assert any("outside Keeravani" in m for m in evaluation.mistakes)


def test_drift_into_a_neighbour_is_named(raagas, mohanam):
    # Mohanam has no madhyamam; using one points somewhere else entirely.
    drifting = notes(mohanam, ["S", "R2", "G3", "P"])
    drifting[3].swara = "M1"
    drifting[3].midi = TONIC + 5
    evaluation = Evaluator(raagas).evaluate(drifting, mohanam)
    assert evaluation.scores["raaga_drift"] < 1.0 or \
        evaluation.scores["swara_correctness"] < 1.0


def test_a_line_that_never_rests_is_marked_down(raagas, keeravani):
    restless = notes(keeravani, ["S", "R2", "G2", "M1", "D1"])
    evaluation = Evaluator(raagas).evaluate(restless, keeravani)
    assert evaluation.scores["structure"] < 0.8
    assert any("rest" in m for m in evaluation.mistakes)


def test_one_note_repeated_is_uninteresting(raagas, keeravani):
    dull = notes(keeravani, ["G2"] * 6)
    evaluation = Evaluator(raagas).evaluate(dull, keeravani)
    assert evaluation.scores["interest"] < 0.5
    assert any("repeats" in m for m in evaluation.mistakes)


def test_ornament_and_dynamics_lift_expressiveness(raagas, keeravani):
    plain = notes(keeravani, ["S", "R2", "G2", "M1", "P", "S"])
    shaped = notes(keeravani, ["S", "R2", "G2", "M1", "P", "S"], gamaka=True)
    shaped[0].velocity, shaped[-1].duration = 105, 1.2
    evaluator = Evaluator(raagas)
    assert (evaluator.evaluate(shaped, keeravani).scores["expressiveness"] >
            evaluator.evaluate(plain, keeravani).scores["expressiveness"])


def test_leaps_everywhere_cost_coherence(raagas, keeravani):
    jumpy = notes(keeravani, ["S", "P", "S-", "N3", "S", "D1-"])
    smooth = notes(keeravani, ["S", "R2", "G2", "M1", "G2", "S"])
    evaluator = Evaluator(raagas)
    assert (evaluator.evaluate(jumpy, keeravani).scores["coherence"] <
            evaluator.evaluate(smooth, keeravani).scores["coherence"])


def test_using_the_raagas_own_phrases_is_rewarded(raagas, keeravani):
    idiom = list(keeravani.prayogas[0])
    evaluator = Evaluator(raagas)
    with_idiom = evaluator.evaluate(notes(keeravani, idiom + ["S"]), keeravani,
                                    learned_phrases=[idiom])
    without = evaluator.evaluate(notes(keeravani, ["S", "M1", "S", "M1", "S"]),
                                 keeravani, learned_phrases=[idiom])
    assert with_idiom.scores["phrase_authenticity"] > \
        without.scores["phrase_authenticity"]


def test_off_grid_rhythm_is_noticed(raagas, keeravani):
    ragged = notes(keeravani, ["S", "R2", "G2", "S"])
    for i, note in enumerate(ragged):
        note.duration = 0.37 + 0.11 * i
    evaluation = Evaluator(raagas).evaluate(ragged, keeravani, tempo_bpm=72)
    assert evaluation.scores["rhythm"] < 0.8


def test_free_rhythm_is_judged_differently(raagas, keeravani):
    breathing = notes(keeravani, ["S", "R2", "G2", "S"])
    for note, length in zip(breathing, (1.6, 0.4, 2.2, 3.0)):
        note.duration = length
    strict = Evaluator(raagas).evaluate(breathing, keeravani, tempo_bpm=72)
    free = Evaluator(raagas).evaluate(breathing, keeravani, tempo_bpm=72,
                                      free_rhythm=True)
    assert free.scores["rhythm"] > strict.scores["rhythm"]


def test_mood_is_matched_against_the_brief(raagas, keeravani, mohanam):
    brief = CreativeBrief(mood="longing", feel="lonely, late at night")
    evaluator = Evaluator(raagas)
    sad = evaluator.evaluate(notes(keeravani, ["S", "R2", "G2", "S"]), keeravani,
                             brief=brief)
    bright = evaluator.evaluate(notes(mohanam, ["S", "R2", "G3", "S"]), mohanam,
                                brief=brief)
    assert sad.scores["mood_match"] >= bright.scores["mood_match"]


def test_the_requested_length_is_checked(raagas, keeravani):
    short = notes(keeravani, ["S", "R2", "G2", "S"], duration=0.5)
    evaluation = Evaluator(raagas).evaluate(short, keeravani,
                                            expected_seconds=60.0)
    assert evaluation.scores["brief_match"] < 0.7
    assert any("asked for" in m for m in evaluation.mistakes)


def test_copied_output_scores_zero_for_originality(raagas, keeravani):
    idiom = ["S", "R2", "G2", "M1", "P", "D1", "N3"]
    index = PhraseIndex(n=4)
    index.add(Phrase(raaga="Keeravani", swaras=idiom))
    evaluation = Evaluator(raagas, index).evaluate(notes(keeravani, idiom),
                                                   keeravani)
    assert evaluation.scores["originality"] < 0.5
    assert evaluation.originality is not None
    assert not evaluation.originality.is_original


def test_nothing_produced_is_reported_clearly(raagas, keeravani):
    evaluation = Evaluator(raagas).evaluate([], keeravani)
    assert evaluation.overall() == 0.0
    assert "nothing was produced" in evaluation.mistakes


def test_the_critic_gives_one_piece_of_advice(raagas, keeravani):
    wrong = notes(keeravani, ["S", "R2", "G2", "M1"])
    wrong[1].swara = "R1"
    wrong[1].midi = TONIC + 1
    evaluation = Evaluator(raagas).evaluate(wrong, keeravani)
    assert evaluation.recommendation
    assert evaluation.weakest(1)[0][1] <= min(evaluation.scores.values()) + 1e-9


def test_confidence_grows_with_evidence(raagas, keeravani):
    evaluator = Evaluator(raagas)
    short = evaluator.evaluate(notes(keeravani, ["S", "R2"]), keeravani)
    long = evaluator.evaluate(
        notes(keeravani, ["S", "R2", "G2", "M1", "P", "D1", "N3", "S+",
                          "N3", "D1", "P", "M1", "S"]), keeravani)
    assert long.confidence > short.confidence
