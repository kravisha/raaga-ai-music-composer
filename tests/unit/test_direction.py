"""The direction reader and its mapping to the generator's controls."""
import pytest

from raagacomposer.agent.guidance import Guidance
from raagacomposer.music.direction import (SectionDirection, apply_direction,
                                           directed_register, read_direction)
from raagacomposer.music.melody import MelodyOptions

pytestmark = pytest.mark.unit


def test_each_control_word_sets_its_control_and_nothing_else():
    cases = {
        "in a closer register": ("register", "closer"),
        "a lower register": ("register", "lower"),
        "take it higher": ("register", "higher"),
        "with fewer leaps": ("motion", "stepwise"),
        "move stepwise": ("motion", "stepwise"),
        "a clearer cadence": ("cadence", "resting"),
        "land on the sa": ("cadence", "resting"),
        "more gamaka": ("ornament", "more"),
        "plainer": ("ornament", "less"),
        "less ornament": ("ornament", "less"),
        "softer": ("energy", "softer"),
        "more inward": ("energy", "softer"),
        "stronger": ("energy", "stronger"),
        "more variation": ("variety", "more"),
    }
    for text, (control, value) in cases.items():
        d = read_direction(text)
        assert getattr(d, control) == value, text
        others = {c for c in ("register", "motion", "cadence", "ornament", "energy", "variety")
                  if getattr(d, c)}
        assert others == {control}, (text, others)
        assert not d.unsupported


def test_a_mixed_sentence_sets_several_and_the_last_word_wins_a_control():
    d = read_direction("give the Charanam a clearer cadence in a closer register, softer")
    assert (d.cadence, d.register, d.energy) == ("resting", "closer", "softer")
    assert d.motion == "" and d.ornament == ""
    d = read_direction("softer - no, stronger")
    assert d.energy == "stronger"


def test_an_emotion_word_is_a_bundle_of_controls_and_says_so():
    d = read_direction("make the Charanam sadder")
    assert (d.energy, d.register, d.cadence) == ("softer", "closer", "resting")
    text = d.describe()
    assert "softer" in text and "closer register" in text and "resting cadence" in text
    assert "sad" not in text.lower().replace("sadder", ""), text
    assert "sadder" in d.words


def test_words_that_name_no_control_are_reported_not_guessed():
    d = read_direction("faster, with more syncopation and a key change")
    assert d.is_empty()
    assert d.unsupported == ["faster", "syncopation", "key change"]
    assert d.describe().startswith("not a control I have: faster")
    d = read_direction("reshape the passage into a closer contour")
    assert d.register == "closer" and not d.unsupported


def test_a_sentence_with_no_property_is_empty():
    d = read_direction("rewrite the Charanam")
    assert d.is_empty() and d.describe() == "" and not d.words


def test_the_register_window_moves_or_narrows_inside_the_voice():
    # closer: a 12-semitone window trimmed 20% each side; the generator
    # then pulls the section's notes inside it by step, never by octave
    assert directed_register(62, 74, 52, 79, "closer") == (64, 72, "")
    lo, hi, why = directed_register(60, 66, 52, 79, "closer")
    assert (lo, hi) == (60, 66) and "within a fifth" in why
    # lower: down a fourth while the voice allows it
    assert directed_register(62, 74, 52, 79, "lower") == (57, 69, "")
    # higher: up a fourth, clipped to the top of the voice
    assert directed_register(62, 74, 52, 79, "higher") == (67, 79, "")
    lo, hi, why = directed_register(52, 64, 52, 79, "lower")
    assert (lo, hi) == (52, 64) and "nothing below" in why
    assert directed_register(62, 74, 52, 79, "") == (62, 74, "")


def test_applying_a_direction_layers_over_the_lesson_guidance_without_lifting_it():
    lessons = Guidance(avoid_swaras={"N3"}, avoid_transitions={("S", "N3")},
                       add_gamaka=True, prefer_step=0.3)
    opts = MelodyOptions(seed=4)
    d = read_direction("plainer, stepwise, with a clearer cadence")
    out = apply_direction(d, opts, lessons)
    assert out.guidance is not lessons and out.opts is not opts
    assert out.guidance.add_gamaka is False, "plainer beats the lesson's soft add_gamaka"
    assert out.guidance.prefer_step == pytest.approx(0.9)
    assert out.guidance.must_end_on_nyasa is True
    assert out.guidance.avoid_swaras == {"N3"} and out.guidance.avoid_transitions == {("S", "N3")}
    assert lessons.add_gamaka is True and lessons.prefer_step == 0.3, "the lessons were changed"
    assert opts.direction is None and out.opts.direction is d
    assert any("forbidden move" in c for c in out.controls)


def test_a_direction_that_cannot_live_with_a_hard_lesson_is_a_conflict():
    class R:
        nyasa = ["S", "P"]
    lessons = Guidance(avoid_endings={"S", "P"})
    out = apply_direction(read_direction("a clearer cadence"), MelodyOptions(), lessons, raaga=R())
    assert out.guidance.must_end_on_nyasa is False
    assert out.conflicts and "resting cadence" in out.conflicts[0]
    out = apply_direction(read_direction("a clearer cadence"), MelodyOptions(),
                          Guidance(avoid_endings={"S"}), raaga=R())
    assert out.guidance.must_end_on_nyasa is True and not out.conflicts


def test_an_empty_direction_changes_nothing():
    lessons = Guidance(add_gamaka=True)
    opts = MelodyOptions(seed=9)
    out = apply_direction(SectionDirection(), opts, lessons)
    assert out.opts.direction is None and out.guidance.add_gamaka is True
    assert not out.controls and not out.conflicts


# --------------------------------------------------------------------------
# Arya's review of 95d89fe: negation, contradiction, correction, the window
# --------------------------------------------------------------------------
def test_a_negated_word_is_declined_not_applied():
    for text in ("not softer", "never louder", "without more gamaka",
                 "keep it as it is, no closer register"):
        d = read_direction(text)
        assert d.is_empty(), text
        assert d.declined, text
        assert "not applied" in d.describe(), text
    d = read_direction("not softer, but plainer")
    assert d.energy == "" and d.ornament == "less" and d.declined == ["softer"]


def test_two_values_for_one_control_are_a_contradiction_and_neither_applies():
    d = read_direction("softer and stronger")
    assert d.energy == "" and d.contradictions == ["softer and stronger"]
    assert "asked both ways" in d.describe()
    d = read_direction("make it lower and higher, plainer")
    assert d.register == "" and d.ornament == "less"
    assert d.contradictions == ["lower and higher"]


def test_an_explicit_correction_is_the_later_word():
    for text in ("softer - no, stronger", "softer, actually stronger",
                 "softer; I mean stronger", "closer, rather lower"):
        d = read_direction(text)
        assert not d.contradictions, text
    assert read_direction("softer - no, stronger").energy == "stronger"
    assert read_direction("closer, rather lower").register == "lower"


def test_closer_never_narrows_below_a_fifth():
    from raagacomposer.music.direction import MIN_WINDOW
    assert MIN_WINDOW == 7
    # eight semitones: one can come off, leaving a fifth... no - a trim of 1
    # each side leaves six, under the fifth, so nothing is trimmed and it
    # is said.
    lo, hi, why = directed_register(60, 68, 52, 79, "closer")
    assert (lo, hi) == (60, 68) and "under a fifth" in why
    for width in (9, 10, 11, 12, 14, 20):
        lo, hi, why = directed_register(60, 60 + width, 52, 90, "closer")
        assert hi - lo >= MIN_WINDOW, (width, lo, hi)
        assert why == "" and hi - lo < width, (width, lo, hi, why)
    lo, hi, why = directed_register(60, 67, 52, 79, "closer")
    assert (lo, hi) == (60, 67) and "within a fifth" in why
